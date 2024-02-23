// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected.hpp"

#include "joint_impl/extended_opset.hpp"
#include "joint_impl/ops/cpu/fully_connected_cpu.hpp"

namespace ov {

FullyConnectedImplementationsRegistry::FullyConnectedImplementationsRegistry() {
    register_impl<cpu::SomeFullyConnectedCPUImpl>();
}

REGISTER_OP(FullyConnected, ov::intel_gpu::op::FullyConnected, FullyConnectedParams, FullyConnectedImplementationsRegistry);

}  // namespace ov
