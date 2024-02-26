// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "joint_impl/layout_optimizer.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {

class LayoutPropagation: public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::intel_gpu::LayoutPropagation");

    explicit LayoutPropagation(std::shared_ptr<const LayoutOptimizer> optimizer)
        : ov::pass::ModelPass()
        , m_optimizer(optimizer) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<const LayoutOptimizer> m_optimizer;
};

}   // namespace ov
