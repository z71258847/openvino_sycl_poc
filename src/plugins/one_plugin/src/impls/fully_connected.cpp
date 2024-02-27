// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected.hpp"

#include "extension/extended_opset.hpp"
#include "backend/cpu/fully_connected_cpu.hpp"

namespace ov {

FullyConnectedImplementationsRegistry::FullyConnectedImplementationsRegistry() {
    register_impl<cpu::SomeFullyConnectedCPUImpl>();
}

REGISTER_IMPLS(FullyConnected, ov::intel_gpu::op::FullyConnected, FullyConnectedParams, FullyConnectedImplementationsRegistry);

}  // namespace ov
