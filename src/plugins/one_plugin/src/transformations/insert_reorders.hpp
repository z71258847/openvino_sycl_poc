// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "execution_config.hpp"
#include "openvino/pass/pass.hpp"
#include "extension/layout_optimizer.hpp"

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
