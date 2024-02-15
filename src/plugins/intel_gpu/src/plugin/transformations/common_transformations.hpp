// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/pass/pass.hpp"

#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/device.hpp"
#include "transformations_context.hpp"

namespace ov {
namespace intel_gpu {

class CommonTransformations : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::intel_gpu::CommonTransformations");
    explicit CommonTransformations(const TransformationsContext& context)
        : m_context(context) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    const TransformationsContext& m_context;
};

}  // namespace intel_gpu
}  // namespace ov
