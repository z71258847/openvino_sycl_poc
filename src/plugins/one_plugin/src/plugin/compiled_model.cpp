// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/except.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/util/common_util.hpp"

#include "compiled_model.hpp"
#include "async_infer_request.hpp"

#include <fstream>
#include <utility>
#include <sys/types.h>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace ov {
namespace intel_gpu {

namespace {
std::shared_ptr<ov::threading::ITaskExecutor> create_task_executor(const std::shared_ptr<const ov::IPlugin>& plugin, const ExecutionConfig& config) {
    if (config.get_property(ov::internal::exclusive_async_requests)) {
        //exclusive_async_requests essentially disables the streams (and hence should be checked first) => aligned with the CPU behavior
        return plugin->get_executor_manager()->get_executor("GPU");
    } else if (config.get_property(ov::hint::enable_cpu_pinning)) {
        auto executor_config =
            ov::threading::IStreamsExecutor::Config{"Intel GPU plugin executor",
                                                    config.get_property(ov::num_streams),
                                                    0,
                                                    ov::threading::IStreamsExecutor::ThreadBindingType::CORES,
                                                    1,
                                                    0,
                                                    0,
                                                    ov::threading::IStreamsExecutor::Config::PreferredCoreType::BIG,
                                                    {{config.get_property(ov::num_streams), MAIN_CORE_PROC, 1, 0, 0}},
                                                    true};
        return std::make_shared<ov::threading::CPUStreamsExecutor>(executor_config);
    } else {
        return std::make_shared<ov::threading::CPUStreamsExecutor>(
            ov::threading::IStreamsExecutor::Config{"Intel GPU plugin executor", config.get_property(ov::num_streams)});
    }
}
}  // namespace

CompiledModel::CompiledModel(std::shared_ptr<ov::Model> model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             RemoteContextImpl::Ptr context,
                             const ExecutionConfig& config)
    : ov::ICompiledModel(model,
                         plugin,
                         context,
                         create_task_executor(plugin, config),
                         nullptr)
    , m_context(context)
    , m_config(config)
    , m_wait_executor(std::make_shared<ov::threading::CPUStreamsExecutor>(ov::threading::IStreamsExecutor::Config{"Intel GPU plugin wait executor"}))
    , m_model_name(model->get_friendly_name())
    , m_inputs(ov::ICompiledModel::inputs())
    , m_outputs(ov::ICompiledModel::outputs())
    , m_loaded_from_cache(false) {
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    auto sync_request = create_sync_infer_request();
    auto async_infer_request = std::make_shared<AsyncInferRequest>(std::static_pointer_cast<SyncInferRequest>(sync_request),
                                                                   get_task_executor(),
                                                                   m_wait_executor,
                                                                   get_callback_executor());
    return async_infer_request;
}

// Cache blob format:
//     [ is_dynamic flag ]
//     [ ov::Node::Input/ ov::Node::Output ]
//     [ ov::intel_gpu::Graph ]
void CompiledModel::export_model(std::ostream& model) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    if (name == ov::supported_properties) {
        return decltype(ov::supported_properties)::value_type {
            // Metrics
            ov::PropertyName{ov::supported_properties.name(), PropertyMutability::RO},
            ov::PropertyName{ov::model_name.name(), PropertyMutability::RO},
            ov::PropertyName{ov::optimal_number_of_infer_requests.name(), PropertyMutability::RO},

            // Configs
            ov::PropertyName{ov::enable_profiling.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::enable_cpu_pinning.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::model_priority.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::hint::host_task_priority.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::hint::queue_priority.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::hint::queue_throttle.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::enable_loop_unrolling.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::disable_winograd_convolution.name(), PropertyMutability::RO},
            ov::PropertyName{ov::cache_dir.name(), PropertyMutability::RO},
            ov::PropertyName{ov::cache_mode.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::performance_mode.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::execution_mode.name(), PropertyMutability::RO},
            ov::PropertyName{ov::compilation_num_threads.name(), PropertyMutability::RO},
            ov::PropertyName{ov::num_streams.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::num_requests.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::inference_precision.name(), PropertyMutability::RO},
            ov::PropertyName{ov::device::id.name(), PropertyMutability::RO},
            ov::PropertyName{ov::execution_devices.name(), PropertyMutability::RO}
        };
    } else if (name == ov::model_name) {
        return decltype(ov::model_name)::value_type {m_model_name};
    } else if (name == ov::loaded_from_cache) {
        return decltype(ov::loaded_from_cache)::value_type {m_loaded_from_cache};
    } else if (name == ov::optimal_number_of_infer_requests) {
        unsigned int nr = m_config.get_property(ov::num_streams);
        if (m_config.get_property(ov::hint::performance_mode) != ov::hint::PerformanceMode::LATENCY)
            nr *= 2;
        return decltype(ov::optimal_number_of_infer_requests)::value_type {nr};
    } else if (name == ov::execution_devices) {
        return decltype(ov::execution_devices)::value_type{m_context->get_device_name()};
    }

    return m_config.get_property(name);
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    return std::make_shared<SyncInferRequest>(std::static_pointer_cast<const CompiledModel>(shared_from_this()));
}

}  // namespace intel_gpu
}  // namespace ov
