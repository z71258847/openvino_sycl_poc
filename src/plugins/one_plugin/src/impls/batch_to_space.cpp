// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_to_space.hpp"
#include "extension/extended_opset.hpp"
#include "openvino/op/batch_to_space.hpp"

#include "backend/cpu/batch_to_space_cpu.hpp"

namespace ov {

BatchToSpaceImplementationsRegistry::BatchToSpaceImplementationsRegistry() {
    register_impl<cpu::SomeBatchToSpaceCPUImpl>();
}

REGISTER_IMPLS(BatchToSpace, ov::op::v1::BatchToSpace, BatchToSpaceParams, BatchToSpaceImplementationsRegistry);

}  // namespace ov
