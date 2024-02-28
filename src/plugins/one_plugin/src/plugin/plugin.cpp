// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "compiled_model.hpp"
#include "execution_config.hpp"
#include "transformations/transformations_pipeline.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/performance_heuristics.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/common_optimizations/dimension_tracking.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/utils/utils.hpp"

// Undef DEVICE_TYPE macro which can be defined somewhere in windows headers as DWORD and conflict with our metric
#ifdef DEVICE_TYPE
#undef DEVICE_TYPE
#endif

using ms = std::chrono::duration<double, std::ratio<1, 1000>>;
using Time = std::chrono::high_resolution_clock;

namespace ov {
namespace intel_gpu {

#define FACTORY_DECLARATION(op_version, op_name) \
    void __register ## _ ## op_name ## _ ## op_version();

#define FACTORY_CALL(op_version, op_name) \
    __register ## _ ## op_name ## _ ## op_version();

std::string Plugin::get_device_id_from_config(const ov::AnyMap& config) const {
    std::string id;
    if (config.find(ov::device::id.name()) != config.end()) {
        id = config.at(ov::device::id.name()).as<std::string>();
    }
    return id;
}

std::string Plugin::get_device_id(const ov::AnyMap& config) const {
    std::string id = m_default_device_id;
    if (config.find(ov::device::id.name()) != config.end()) {
        id = config.at(ov::device::id.name()).as<std::string>();
    }
    return id;
}

void Plugin::transform_model(std::shared_ptr<ov::Model>& model, const ExecutionConfig& config) const {
    // auto deviceInfo = m_device_map.at(config.get_property(ov::device::id))->get_info();
    TransformationsPipeline transformations(config/* , deviceInfo */);
    transformations.run_on_model(model);
}

std::shared_ptr<ov::Model> Plugin::clone_and_transform_model(const std::shared_ptr<const ov::Model>& model, const ExecutionConfig& config) const {
    auto cloned_model = model->clone();
    OPENVINO_ASSERT(cloned_model != nullptr, "[GPU] Failed to clone model!");

    transform_model(cloned_model, config);

    // Transformations for some reason may drop output tensor names, so here we copy those from the original model
    auto new_results = cloned_model->get_results();
    auto old_results = model->get_results();
    OPENVINO_ASSERT(new_results.size() == old_results.size(), "[GPU] Unexpected outputs count change in transformed model",
                                                              "Before: ", old_results.size(), " After: ", new_results.size());
    for (size_t i = 0; i < model->get_results().size(); i++) {
        auto new_res = new_results[i];
        auto old_res = old_results[i];

        new_res->output(0).set_names(old_res->output(0).get_names());
        new_res->set_friendly_name(old_res->get_friendly_name());
    }

    return cloned_model;
}

std::map<std::string, RemoteContextImpl::Ptr> Plugin::get_default_contexts() const {
    std::call_once(m_default_contexts_once, [this]() {
        // Create default context
        // for (auto& device : m_device_map) {
        //     auto ctx = std::make_shared<RemoteContextImpl>(get_device_name() + "." + device.first, std::vector<cldnn::device::ptr>{ device.second });
        //     m_default_contexts.insert({device.first, ctx});
        // }
    });
    return m_default_contexts;
}

Plugin::Plugin() {
    set_device_name("ONE");

    // Set OCL runtime which should be always available
    // cldnn::device_query device_query(cldnn::engine_types::ocl, cldnn::runtime_types::ocl);
    // m_device_map = device_query.get_available_devices();

    // // Set default configs for each device
    // for (const auto& device : m_device_map) {
    //     m_configs_map.insert({device.first, ExecutionConfig(ov::device::id(device.first))});
    // }

    // Set common info for compiled_model_runtime_properties
    auto& ov_version = ov::get_openvino_version();
    m_compiled_model_runtime_properties["OV_VERSION"] = ov_version.buildNumber;
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model, const ov::AnyMap& orig_config) const {
    std::string device_id = get_device_id(orig_config);

    auto context = get_default_context(device_id);

    // OPENVINO_ASSERT(m_configs_map.find(device_id) != m_configs_map.end(), "[GPU] compile_model: Couldn't find config for GPU with id ", device_id);

    ExecutionConfig config;// = m_configs_map.at(device_id);
    config.set_user_property(orig_config);
    config.apply_user_properties(/* context->get_engine().get_device_info() */);

    auto transformed_model = clone_and_transform_model(model, config);
    return std::make_shared<CompiledModel>(transformed_model, shared_from_this(), context, config);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& orig_config,
                                                          const ov::SoPtr<ov::IRemoteContext>& context) const {
    auto context_impl = get_context_impl(context);
    auto device_id = ov::DeviceIDParser{context_impl->get_device_name()}.get_device_id();

    // OPENVINO_ASSERT(m_configs_map.find(device_id) != m_configs_map.end(), "[GPU] LoadExeNetworkImpl: Couldn't find config for GPU with id ", device_id);

    ExecutionConfig config;// = m_configs_map.at(device_id);
    config.set_user_property(orig_config);
    config.apply_user_properties(/* context_impl->get_engine().get_device_info() */);

    auto transformed_model = clone_and_transform_model(model, config);
    return std::make_shared<CompiledModel>(transformed_model, shared_from_this(), context_impl, config);
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap& remote_properties) const {
    if (remote_properties.empty()) {
        return get_default_context(m_default_device_id);
    }
    return std::make_shared<RemoteContextImpl>(get_default_contexts(), remote_properties);
}

std::shared_ptr<RemoteContextImpl> Plugin::get_default_context(const std::string& device_id) const {
    return nullptr;
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const AnyMap& params) const {
    std::string device_id = m_default_device_id;

    if (params.find(ov::device::id.name()) != params.end())
        device_id = params.at(ov::device::id.name()).as<std::string>();

    return get_default_context(device_id);
}

void Plugin::set_property(const ov::AnyMap &config) {
    // auto update_config = [](ExecutionConfig& config, const ov::AnyMap& user_config) {
    //     config.set_user_property(user_config);
    //     // Check that custom layers config can be loaded
    //     // if (user_config.find(ov::intel_gpu::config_file.name()) != user_config.end()) {
    //     //     CustomLayerMap custom_layers;
    //     //     auto custom_layers_config = user_config.at(ov::intel_gpu::config_file.name()).as<std::string>();
    //     //     CustomLayer::LoadFromFile(custom_layers_config, custom_layers, custom_layers_config.empty());
    //     // }
    // };

    // if (config.find(ov::internal::config_device_id.name()) != config.end()) {
    //     std::string device_id = config.at(ov::internal::config_device_id.name()).as<std::string>();
    //     auto config_for_device = config;
    //     config_for_device.erase(ov::internal::config_device_id.name());
    //     update_config(m_configs_map.at(device_id), config_for_device);
    // } else {
    //     std::string device_id = get_device_id_from_config(config);
    //     if (!device_id.empty()) {
    //         m_default_device_id = device_id;
    //         update_config(m_configs_map.at(device_id), config);
    //     } else {
    //         for (auto& conf : m_configs_map) {
    //             update_config(conf.second, config);
    //         }
    //     }
    // }
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model, const ov::AnyMap& orig_config) const {
    ov::SupportedOpsMap res;
    // std::string device_id = get_device_id(orig_config);

    // auto ctx = get_default_context(device_id);

    // ExecutionConfig config = m_configs_map.at(device_id);
    // config.set_user_property(orig_config);
    // config.apply_user_properties(ctx->get_engine().get_device_info());

    // ProgramBuilder prog(ctx->get_engine(), config);

    // auto supported = ov::get_supported_nodes(model,
    //     [&config,this](std::shared_ptr<ov::Model>& model) {
    //         std::map<std::string, ov::PartialShape> shapes;
    //         std::map<std::string, std::pair<int64_t, int64_t>> batch_dim;
    //         transform_model(model, config);
    //     },
    //     [&prog](std::shared_ptr<ov::Node> node) {
    //         return prog.is_op_supported(node);
    //     });

    // for (auto&& op_name : supported) {
    //     res.emplace(op_name, ctx->get_device_name());
    // }

    return res;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model, const ov::AnyMap& config) const {
    std::string device_id = get_device_id(config);
    auto context = get_default_context(device_id);
    return import_model(model, { context, nullptr }, config);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& orig_config) const {

    // auto context_impl = get_context_impl(context);
    // auto device_id = ov::DeviceIDParser{context_impl->get_device_name()}.get_device_id();

    // // check ov::loaded_from_cache property and erase it due to not needed any more.
    // auto _orig_config = orig_config;
    // const auto& it = _orig_config.find(ov::loaded_from_cache.name());
    // bool loaded_from_cache = false;
    // if (it != _orig_config.end()) {
    //     loaded_from_cache = it->second.as<bool>();
    //     _orig_config.erase(it);
    // }

    // ExecutionConfig config = m_configs_map.at(device_id);
    // config.set_user_property(_orig_config);
    // config.apply_user_properties(context_impl->get_engine().get_device_info());

    // if (config.get_property(ov::cache_mode) == ov::CacheMode::OPTIMIZE_SIZE)
    //     return nullptr;

    // cldnn::BinaryInputBuffer ib(model, context_impl->get_engine());
    // return std::make_shared<CompiledModel>(ib, shared_from_this(), context_impl, config, loaded_from_cache);
    return nullptr;
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& options) const {

    // The metrics below don't depend on the device ID, so we should handle those
    // earler than querying actual ID to avoid exceptions when no devices are found
    if (name == ov::supported_properties) {
        return decltype(ov::supported_properties)::value_type {get_supported_properties()};
    } else if (ov::internal::supported_properties == name) {
        return decltype(ov::internal::supported_properties)::value_type{get_supported_internal_properties()};
    } else if (name == ov::available_devices) {
        std::vector<std::string> available_devices = { "ONE "};
        // for (auto const& dev : m_device_map)
        //     available_devices.push_back(dev.first);
        return decltype(ov::available_devices)::value_type {available_devices};
    } else if (name == ov::internal::caching_properties) {
        return decltype(ov::internal::caching_properties)::value_type(get_caching_properties());
    }

    // ov::AnyMap actual_runtime_info;
    // auto prepare_actual_runtime_info = [&]() {
    //     // Suppose all devices share the same version driver.
    //     auto device_id = m_default_device_id;
    //     OPENVINO_ASSERT(m_device_map.find(device_id) != m_device_map.end(),
    //                     "[GPU] compiled_model_runtime_properties: Couldn't find device for GPU with id ",
    //                     device_id);
    //     actual_runtime_info["DRIVER_VERSION"] = m_device_map.at(device_id)->get_info().driver_version;
    //     // More items can be inserted if needed
    // };
    // Below properties depend on the device ID.
    // if (name == ov::internal::compiled_model_runtime_properties.name()) {
    //     prepare_actual_runtime_info();
    //     auto model_runtime_info = m_compiled_model_runtime_properties;
    //     // Set specified device info for compiled_model_runtime_properties
    //     model_runtime_info.insert(actual_runtime_info.begin(), actual_runtime_info.end());
    //     auto model_format = ov::Any(model_runtime_info);
    //     return decltype(ov::internal::compiled_model_runtime_properties)::value_type(
    //         std::move(model_format.as<std::string>()));
    // } else if (name == ov::internal::compiled_model_runtime_properties_supported.name()) {
    //     ov::Any res = true;
    //     prepare_actual_runtime_info();
    //     auto it = options.find(ov::internal::compiled_model_runtime_properties.name());
    //     if (it == options.end()) {
    //         res = false;
    //         return res;
    //     }
    //     ov::AnyMap input_map = it->second.as<ov::AnyMap>();
    //     // Check common info of compiled_model_runtime_properties
    //     for (auto& item : m_compiled_model_runtime_properties) {
    //         auto it = input_map.find(item.first);
    //         if (it == input_map.end() || it->second.as<std::string>() != item.second.as<std::string>()) {
    //             res = false;
    //             return res;
    //         }
    //     }
    //     // Check specified device info of compiled_model_runtime_properties
    //     for (const auto& it : actual_runtime_info) {
    //         auto item = input_map.find(it.first);
    //         if (item == input_map.end() || item->second.as<std::string>() != it.second.as<std::string>()) {
    //             res = false;
    //             break;
    //         }
    //     }
    //     return res;
    // }

    // OPENVINO_ASSERT(!m_device_map.empty(), "[GPU] Can't get ", name, " property as no supported devices found or an error happened during devices query.\n"
    //                                        "[GPU] Please check OpenVINO documentation for GPU drivers setup guide.\n");

    if (is_metric(name)) {
        return get_metric(name, options);
    }

    std::string device_id = m_default_device_id;
    if (options.find(ov::device::id.name()) != options.end()) {
        device_id = options.find(ov::device::id.name())->second.as<std::string>();
    }

    ExecutionConfig c;
    // OPENVINO_ASSERT(m_configs_map.find(device_id) != m_configs_map.end(), "[GPU] get_property: Couldn't find config for GPU with id ", device_id);

    // const auto& c = m_configs_map.at(device_id);
    return c.get_property(name);
}

auto StringRightTrim = [](std::string string, std::string substring, bool case_sensitive = true) {
    auto ret_str = string;
    if (!case_sensitive) {
        std::transform(string.begin(), string.end(), string.begin(), ::tolower);
        std::transform(substring.begin(), substring.end(), substring.begin(), ::tolower);
    }
    auto erase_position = string.rfind(substring);
    if (erase_position != std::string::npos) {
        // if space exists before substring remove it also
        if (std::isspace(string.at(erase_position - 1))) {
            erase_position--;
        }
        return ret_str.substr(0, erase_position);
    }
    return ret_str;
};

bool Plugin::is_metric(const std::string& name) const {
    // auto all_properties = get_supported_properties();
    // auto internal_properties = get_supported_internal_properties();
    // auto caching_properties = get_caching_properties();
    // all_properties.insert(all_properties.end(), internal_properties.begin(), internal_properties.end());
    // all_properties.insert(all_properties.end(), caching_properties.begin(), caching_properties.end());
    // auto it = std::find(all_properties.begin(), all_properties.end(), name);
    // OPENVINO_ASSERT(it != all_properties.end(), "[GPU] Property ", name, " is not in a list of supported properties");

    // return !it->is_mutable();
    return false;
}

ov::Any Plugin::get_metric(const std::string& name, const ov::AnyMap& options) const {
    // GPU_DEBUG_GET_INSTANCE(debug_config);

    // auto device_id = get_property(ov::device::id.name(), options).as<std::string>();

    // auto iter = m_device_map.find(std::to_string(cldnn::device_query::device_id));
    // if (iter == m_device_map.end())
    //     iter = m_device_map.find(device_id);
    // if (iter == m_device_map.end())
    //     iter = m_device_map.begin();
    // auto device = iter->second;
    // auto device_info = device->get_info();

    // if (name == ov::intel_gpu::device_total_mem_size) {
    //     return decltype(ov::intel_gpu::device_total_mem_size)::value_type {device_info.max_global_mem_size};
    // } else if (name == ov::device::type) {
    //     auto dev_type = device_info.dev_type == cldnn::device_type::discrete_gpu ? ov::device::Type::DISCRETE : ov::device::Type::INTEGRATED;
    //     return decltype(ov::device::type)::value_type {dev_type};
    // } else if (name == ov::device::gops) {
    //     std::map<element::Type, float> gops;
    //     gops[element::i8] = device->get_gops(cldnn::data_types::i8);
    //     gops[element::u8] = device->get_gops(cldnn::data_types::u8);
    //     gops[element::f16] = device->get_gops(cldnn::data_types::f16);
    //     gops[element::f32] = device->get_gops(cldnn::data_types::f32);
    //     return decltype(ov::device::gops)::value_type {gops};
    // } else if (name == ov::intel_gpu::execution_units_count) {
    //     return static_cast<decltype(ov::intel_gpu::execution_units_count)::value_type>(device_info.execution_units_count);
    // } else if (name == ov::intel_gpu::uarch_version) {
    //     std::stringstream s;
    //     if (device_info.gfx_ver.major == 0 && device_info.gfx_ver.minor == 0 && device_info.gfx_ver.revision == 0) {
    //         s << "unknown";
    //     } else {
    //         s << static_cast<int>(device_info.gfx_ver.major) << "."
    //           << static_cast<int>(device_info.gfx_ver.minor) << "."
    //           << static_cast<int>(device_info.gfx_ver.revision);
    //     }
    //     return decltype(ov::intel_gpu::uarch_version)::value_type {s.str()};
    // } else if (name == ov::optimal_batch_size) {
    //     return decltype(ov::optimal_batch_size)::value_type {get_optimal_batch_size(options)};
    // } else if (name == ov::device::uuid) {
    //     return decltype(ov::device::uuid)::value_type {device_info.uuid};
    // } else if (name == ov::device::luid) {
    //     return decltype(ov::device::luid)::value_type {device_info.luid};
    // } else if (name == ov::device::full_name) {
    //     auto deviceName = StringRightTrim(device_info.dev_name, "NEO", false);
    //     deviceName += std::string(" (") + (device_info.dev_type == cldnn::device_type::discrete_gpu ? "dGPU" : "iGPU") + ")";
    //     return decltype(ov::device::full_name)::value_type {deviceName};
    // } else if (name == ov::device::capabilities) {
    //     return decltype(ov::device::capabilities)::value_type {get_device_capabilities(device_info)};
    // } else if (name == ov::range_for_async_infer_requests) {
    //     std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 2, 1);
    //     return decltype(ov::range_for_async_infer_requests)::value_type {range};
    // } else if (name == ov::range_for_streams) {
    //     std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, device_info.num_ccs == 1 ? 2 : device_info.num_ccs);
    //     return decltype(ov::range_for_streams)::value_type {range};
    // } else if (name == ov::intel_gpu::memory_statistics) {
    //     const auto& ctx = get_default_context(device_id);
    //     return decltype(ov::intel_gpu::memory_statistics)::value_type {ctx->get_engine().get_memory_statistics()};
    // } else if (name == ov::max_batch_size) {
    //     return decltype(ov::max_batch_size)::value_type {get_max_batch_size(options)};
    // } else if (name == ov::intel_gpu::driver_version) {
    //     return decltype(ov::intel_gpu::driver_version)::value_type {device_info.driver_version};
    // } else if (name == ov::intel_gpu::device_id) {
    //     std::stringstream s;
    //     s << "0x" << std::hex << device_info.device_id;
    //     return decltype(ov::intel_gpu::device_id)::value_type {s.str()};
    // } else if (name == ov::device::architecture) {
    //     std::stringstream s;
    //     s << "GPU: vendor=0x" << std::hex << device_info.vendor_id << std::dec << " arch=";
    //     if (device_info.gfx_ver.major == 0 && device_info.gfx_ver.minor == 0) {
    //         s << device_info.dev_name;
    //     } else {
    //         s << "v" << static_cast<int>(device_info.gfx_ver.major)
    //           << "." << static_cast<int>(device_info.gfx_ver.minor)
    //           << "." << static_cast<int>(device_info.gfx_ver.revision);
    //     }
    //     return decltype(ov::device::architecture)::value_type {s.str()};
    // } else {
    //     OPENVINO_THROW("Unsupported metric key ", name);
    // }
    OPENVINO_THROW("Unsupported metric key ", name);
}

std::vector<ov::PropertyName> Plugin::get_caching_properties() const {
    static const std::vector<ov::PropertyName> caching_properties =  {
        ov::PropertyName{ov::device::architecture.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::execution_units_count.name(), PropertyMutability::RO},
        ov::PropertyName{ov::hint::inference_precision.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::execution_mode.name(), PropertyMutability::RW},
    };

    return caching_properties;
}

std::vector<ov::PropertyName> Plugin::get_supported_properties() const {
    static const std::vector<ov::PropertyName> supported_properties = {
        // Metrics
        ov::PropertyName{ov::supported_properties.name(), PropertyMutability::RO},
        ov::PropertyName{ov::available_devices.name(), PropertyMutability::RO},
        ov::PropertyName{ov::range_for_async_infer_requests.name(), PropertyMutability::RO},
        ov::PropertyName{ov::range_for_streams.name(), PropertyMutability::RO},
        ov::PropertyName{ov::optimal_batch_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::max_batch_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::architecture.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::full_name.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::uuid.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::luid.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::type.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::gops.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::capabilities.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::device_total_mem_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::uarch_version.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::execution_units_count.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::memory_statistics.name(), PropertyMutability::RO},

        // Configs
        ov::PropertyName{ov::enable_profiling.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::model_priority.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::host_task_priority.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::queue_priority.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::queue_throttle.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::enable_loop_unrolling.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::disable_winograd_convolution.name(), PropertyMutability::RW},
        ov::PropertyName{ov::cache_dir.name(), PropertyMutability::RW},
        ov::PropertyName{ov::cache_mode.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::performance_mode.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::execution_mode.name(), PropertyMutability::RW},
        ov::PropertyName{ov::compilation_num_threads.name(), PropertyMutability::RW},
        ov::PropertyName{ov::num_streams.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::num_requests.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::inference_precision.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::enable_cpu_pinning.name(), PropertyMutability::RW},
        ov::PropertyName{ov::device::id.name(), PropertyMutability::RW},
    };

    return supported_properties;
}

std::vector<ov::PropertyName> Plugin::get_supported_internal_properties() const {
    static const std::vector<ov::PropertyName> supported_internal_properties = {
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::config_device_id.name(), ov::PropertyMutability::WO},
            ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::internal::compiled_model_runtime_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::compiled_model_runtime_properties_supported.name(), ov::PropertyMutability::RO}};
    return supported_internal_properties;
}

// std::vector<std::string> Plugin::get_device_capabilities(const cldnn::device_info& info) const {
//     std::vector<std::string> capabilities;

//     capabilities.emplace_back(ov::device::capability::FP32);
//     capabilities.emplace_back(ov::device::capability::BIN);
//     if (info.supports_fp16)
//         capabilities.emplace_back(ov::device::capability::FP16);
//     if (info.supports_imad || info.supports_immad)
//         capabilities.emplace_back(ov::device::capability::INT8);
//     if (info.supports_immad)
//         capabilities.emplace_back(ov::intel_gpu::capability::HW_MATMUL);
//     capabilities.emplace_back(ov::device::capability::EXPORT_IMPORT);

//     return capabilities;
// }

uint32_t Plugin::get_max_batch_size(const ov::AnyMap& options) const {
    return 1;
}

uint32_t Plugin::get_optimal_batch_size(const ov::AnyMap& options) const {
    return 1;
}

}  // namespace intel_gpu
}  // namespace ov

static const ov::Version version = { CI_BUILD_NUMBER, "Intel GPU plugin" };
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::intel_gpu::Plugin, version)
