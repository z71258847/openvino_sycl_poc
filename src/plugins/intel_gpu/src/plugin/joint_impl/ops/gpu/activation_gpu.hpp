// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "joint_impl/ops/activation.hpp"

namespace ov {
namespace gpu {

class SomeActivationGPUImpl : public OpImplementation {
public:
    SomeActivationGPUImpl(const ActivationParams& params) : OpImplementation("SomeActivationGPUImpl"), m_params(params) {}

    void execute() override;
    const ActivationParams& m_params;
};

}  // namespace gpu
}  // namespace ov
