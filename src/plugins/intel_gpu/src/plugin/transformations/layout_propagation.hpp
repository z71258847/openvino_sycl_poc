// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "layout_optimizer.hpp"

namespace ov {
namespace intel_gpu {

class LayoutPropagation: public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::intel_gpu::LayoutPropagation");

    explicit LayoutPropagation(const LayoutOptimizer& optimizer)
        : ov::pass::ModelPass()
        , m_optimizer(optimizer) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    const LayoutOptimizer& m_optimizer;
};

}   // namespace intel_gpu
}   // namespace ov
