// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation.hpp"
#include "extension/extended_opset.hpp"

#include "backend/cpu/activation_cpu.hpp"
#include "backend/ocl/activation_ocl.hpp"


namespace ov {

ActivationImplementationsRegistry::ActivationImplementationsRegistry() {
    register_impl<cpu::SomeActivationCPUImpl>();
    register_impl<ocl::SomeActivationOCLImpl>();
}

REGISTER_IMPLS(Abs, ov::op::v0::Abs, ActivationParams, ActivationImplementationsRegistry);
REGISTER_IMPLS(Relu, ov::op::v0::Relu, ActivationParams, ActivationImplementationsRegistry);

}  // namespace ov
