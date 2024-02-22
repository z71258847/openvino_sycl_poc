// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "joint_impl/ops/activation.hpp"

namespace ov {
namespace cpu {

class SomeActivationCPUImpl : public OpImplementation {
public:
    SomeActivationCPUImpl(const ActivationParams& params) : OpImplementation("SomeActivationCPUImpl"), m_params(params) {}

    void execute() override;
    const ActivationParams& m_params;
};

}  // namespace cpu
}  // namespace ov
