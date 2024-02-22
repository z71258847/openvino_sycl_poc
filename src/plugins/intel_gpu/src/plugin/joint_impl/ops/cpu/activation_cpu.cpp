// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_cpu.hpp"

namespace ov {
namespace cpu {

void SomeActivationCPUImpl::execute() {
    std::cerr << "SomeActivationCPUImpl::execute()" << (int)m_params.type << "\n";
}

}  // namespace cpu
}  // namespace ov
