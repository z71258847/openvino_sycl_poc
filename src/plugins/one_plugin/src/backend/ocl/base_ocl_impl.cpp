// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base_ocl_impl.hpp"

namespace ov {
namespace ocl {

const KernelsDataBase BaseOCLImpl::m_db;

void BaseOCLImpl::initialize(const ImplementationParameters* params) {
    m_params = params;
    init_kernel_data(m_params);
}

}  // namespace ocl
}  // namespace ov
