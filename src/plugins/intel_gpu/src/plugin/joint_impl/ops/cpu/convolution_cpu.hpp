// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "joint_impl/ops/convolution.hpp"

namespace ov {
namespace cpu {

class SomeConvolutionCPUImpl : public OpImplementation {
public:
    SomeConvolutionCPUImpl(const SomeCustomParams& params) : OpImplementation("SomeConvolutionCPUImpl") {}
    void execute() override;
};

}  // namespace cpu
}  // namespace ov
