// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_gpu.hpp"

namespace ov {
namespace gpu {

void SomeActivationGPUImpl::execute() {
    std::cerr << "SomeActivationGPUImpl::execute()" << (int)m_params.type << "\n";
}

}  // namespace gpu
}  // namespace ov
