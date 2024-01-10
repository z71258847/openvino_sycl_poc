// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "openvino/pass/pass.hpp"
#include "layout_optimizer.hpp"

using DeviceInfo = cldnn::device_info;

namespace ov {
namespace intel_gpu {

class InsertReorders: public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::intel_gpu::LayoutPropagation");

    InsertReorders(const LayoutOptimizer& optimizer)
        : ov::pass::ModelPass()
        , m_optimizer(optimizer) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    const LayoutOptimizer& m_optimizer;
};

}   // namespace intel_gpu
}   // namespace ov
