// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations_context.hpp"
#include "low_precision/low_precision.hpp"

namespace ov {
namespace intel_gpu {

TransformationsContext::TransformationsContext(std::shared_ptr<const ov::Model> model, const ExecutionConfig& config, const DeviceInfo& device_info)
    : m_config(config)
    , m_device_info(device_info) {
    m_is_model_quantized = ov::pass::low_precision::LowPrecision::isFunctionQuantized(model);
}

bool TransformationsContext::run_lpt() const {
    return m_is_model_quantized && m_config.get_property(ov::intel_gpu::enable_lp_transformations);
}

bool TransformationsContext::has_dpas() const {
    return m_device_info.supports_immad;
}

bool TransformationsContext::unroll_loop() const {
    return m_config.get_property(ov::intel_gpu::enable_loop_unrolling);
}

bool TransformationsContext::is_model_quantized() const {
    return m_is_model_quantized;
}

}  // namespace intel_gpu
}  // namespace ov
