// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "joint_impl/ops/activation.hpp"

#include "joint_impl/ops/cpu/activation_cpu.hpp"
#include "joint_impl/ops/gpu/activation_gpu.hpp"

#include "joint_impl/extended_opset.hpp"

namespace ov {

ActivationImplementationsRegistry::ActivationImplementationsRegistry() {
    register_impl<cpu::SomeActivationCPUImpl>();
    register_impl<gpu::SomeActivationGPUImpl>();
}

REGISTER_IMPLS(Abs, ov::op::v0::Abs, ActivationParams, ActivationImplementationsRegistry);
REGISTER_IMPLS(Relu, ov::op::v0::Relu, ActivationParams, ActivationImplementationsRegistry);

}  // namespace ov
