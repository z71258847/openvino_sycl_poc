// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/pass/pass.hpp"

#include "transformations_context.hpp"

namespace ov {
namespace intel_gpu {

class OptimizeWithInternalOpset : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::intel_gpu::OptimizeWithInternalOpset");
    explicit OptimizeWithInternalOpset(const TransformationsContext& context)
        : m_context(context) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    const TransformationsContext& m_context;
};

}  // namespace intel_gpu
}  // namespace ov
