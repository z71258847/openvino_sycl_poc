// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"
#include "extension/extended_opset.hpp"
#include "backend/cpu/eltwise_cpu.hpp"

namespace ov {

EltwiseRegistry::EltwiseRegistry() {
    register_impl<cpu::SomeEltwiseCPUImpl>();
}

REGISTER_IMPLS(Add, op::v1::Add, EltwiseParams, EltwiseRegistry);
REGISTER_IMPLS(Subtract, op::v1::Subtract, EltwiseParams, EltwiseRegistry);

}  // namespace ov
