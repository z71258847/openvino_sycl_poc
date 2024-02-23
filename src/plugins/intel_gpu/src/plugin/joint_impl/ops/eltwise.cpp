// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"
#include "joint_impl/extended_opset.hpp"
#include "joint_impl/ops/cpu/eltwise_cpu.hpp"

namespace ov {

EltwiseRegistry::EltwiseRegistry() {
    register_impl<cpu::SomeEltwiseCPUImpl>();
}

REGISTER_OP(Add, op::v1::Add, EltwiseParams, EltwiseRegistry);
REGISTER_OP(Subtract, op::v1::Subtract, EltwiseParams, EltwiseRegistry);

}  // namespace ov
