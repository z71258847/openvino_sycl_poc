// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/device.hpp"

namespace ov {
namespace intel_gpu {

class TransformationsContext {
public:
    TransformationsContext(std::shared_ptr<const ov::Model> model, const ExecutionConfig& config, const DeviceInfo& device_info);

    const ExecutionConfig& get_config() const { return m_config; }
    const DeviceInfo& get_device_info() const { return m_device_info; }

    bool run_lpt() const;
    bool has_dpas() const;
    bool unroll_loop() const;
    bool is_model_quantized() const;

private:
    const ExecutionConfig& m_config;
    const DeviceInfo& m_device_info;
    bool m_is_model_quantized = false;
};

}  // namespace intel_gpu
}  // namespace ov
