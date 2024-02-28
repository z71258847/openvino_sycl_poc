// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base_cpu_impl.hpp"

namespace ov {
namespace cpu {

void BaseCPUImpl::initialize(const ImplementationParameters* params) {
    m_params = params;
}

}  // namespace cpu
}  // namespace ov
