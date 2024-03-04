// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape.hpp"
#include "extension/extended_opset.hpp"
#include "backend/cpu/reshape_cpu.hpp"

namespace ov {

ReshapeRegistry::ReshapeRegistry() {
    register_impl<cpu::SomeReshapeCPUImpl>();
}

REGISTER_IMPLS(Reshape, op::v1::Reshape, ReshapeParams, ReshapeRegistry);

}  // namespace ov
