// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_ocl.hpp"
// #include "activation/activation_kernel_base.h"
// #include "common_types.h"
// #include "impls/ocl/kernel_selector_helper.h"
// #include "intel_gpu/graph/program.hpp"
// #include "intel_gpu/primitives/activation.hpp"
// #include "intel_gpu/runtime/device_query.hpp"
// #include "intel_gpu/runtime/engine.hpp"
// #include "intel_gpu/runtime/engine_configuration.hpp"
// #include "intel_gpu/runtime/execution_config.hpp"
#include "extension/executor.hpp"
#include "impls/activation.hpp"

// #include "kernel_selector/kernels/activation/activation_kernel_selector.h"
// #include "kernel_selector_params.h"
#include "openvino/core/partial_shape.hpp"

namespace {

// void set_params(const cldnn::engine& engine, const ov::intel_gpu::ExecutionConfig& config, kernel_selector::params& params) {

//     const auto& device_info = engine.get_device_info();

//     params.uniqueID = std::to_string(100500);
//     params.engineInfo.supports_fp16 = device_info.supports_fp16;
//     params.engineInfo.supports_fp64 = device_info.supports_fp64;
//     params.engineInfo.supports_fp16_denorms = device_info.supports_fp16_denorms;
//     params.engineInfo.supports_khr_subgroups = device_info.supports_khr_subgroups;
//     params.engineInfo.supports_intel_subgroups = device_info.supports_intel_subgroups;
//     params.engineInfo.supports_intel_subgroups_short = device_info.supports_intel_subgroups_short;
//     params.engineInfo.supports_intel_subgroups_char = device_info.supports_intel_subgroups_char;
//     params.engineInfo.supports_intel_required_subgroup_size = device_info.supports_intel_required_subgroup_size;
//     params.engineInfo.supports_image = device_info.supports_image;

//     params.engineInfo.supports_imad = device_info.supports_imad;
//     params.engineInfo.supports_immad = device_info.supports_immad;
//     params.engineInfo.enable_sub_groups_emulation = true;
//     params.engineInfo.bOptHintsSupport = false;

//     params.engineInfo.bLocalBlockIOSupport = false;
//     params.engineInfo.deviceType = kernel_selector::dev_type::integrated_gpu;
//     params.engineInfo.maxWorkGroupSize = device_info.max_work_group_size;
//     params.engineInfo.maxLocalMemSize = device_info.max_local_mem_size;
//     params.engineInfo.maxImage2dWidth = device_info.max_image2d_width;
//     params.engineInfo.maxImage2dHeight = device_info.max_image2d_height;
//     params.engineInfo.computeUnitsCount = device_info.execution_units_count;
//     params.engineInfo.maxThreadsPerExecutionUnit = device_info.num_threads_per_eu > 0 ? device_info.num_threads_per_eu : 7;
//     params.engineInfo.maxThreadsPerDevice = params.engineInfo.maxThreadsPerExecutionUnit * device_info.execution_units_count;
//     params.engineInfo.driverVersion = device_info.driver_version;
//     params.engineInfo.supportedSimdSizes = device_info.supported_simd_sizes;
//     params.engineInfo.vendor_id = device_info.vendor_id;

//     auto impl_forcing = config.get_property(ov::intel_gpu::force_implementations);

//     params.allowStaticInputReordering = config.get_property(ov::intel_gpu::optimize_data) || config.get_property(ov::intel_gpu::allow_static_input_reorder);
//     params.allowInputReordering = false;
// }

}

namespace ov {
namespace ocl {

class SomeActivationOCLExecutor : public OpExecutor {
public:
    explicit SomeActivationOCLExecutor(const ActivationParams* params) : m_params(params) {

    }

    void execute() override {
        std::cerr << "SomeActivationOCLExecutor::execute()" << (int)m_params->type << "\n";
    }

private:
    const ActivationParams* m_params;
};


bool SomeActivationOCLImpl::supports(const ImplementationParameters* params) const {
    return true;
}

OpExecutor::Ptr SomeActivationOCLImpl::get_executor(const ImplementationParameters* params) const {
    auto typed_params = dynamic_cast<const ActivationParams*>(params);

    // using namespace kernel_selector;

    // activation_kernel_selector selector;
    // activation_params selector_params;


    // auto devices = cldnn::device_query(cldnn::engine_types::ocl, cldnn::runtime_types::ocl).get_available_devices();
    // auto engine = cldnn::engine::create(cldnn::engine_types::ocl, cldnn::runtime_types::ocl, devices.begin()->second);
    // intel_gpu::ExecutionConfig cfg;
    // set_params(*engine, cfg, selector_params);
    // selector_params.inputs.resize(1);
    // selector_params.outputs.resize(1);

    // selector_params.inputs[0] = cldnn::convert_data_tensor(cldnn::layout(ov::PartialShape{1, 2, 3, 4}, ov::element::f32, cldnn::format::bfyx));
    // selector_params.outputs[0] = cldnn::convert_data_tensor(cldnn::layout(ov::PartialShape{1, 2, 3, 4}, ov::element::f32, cldnn::format::bfyx));

    // if (typed_params->type == ActivationParams::Type::Abs) {
    //     selector_params.activations.push_back(base_activation_params{ActivationFunction::ABS, 0.0f, 0.0f});
    // } else if (typed_params->type == ActivationParams::Type::ReLU) {
    //     selector_params.activations.push_back(base_activation_params{ActivationFunction::RELU, 0.0f, 0.0f});
    // }

    // auto kernel = selector.get_best_kernel(selector_params);

    return std::make_shared<SomeActivationOCLExecutor>(typed_params);
}

}  // namespace ocl
}  // namespace ov
