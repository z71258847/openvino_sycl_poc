// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend/ocl/common/kernel_data.hpp"
#include "backend/ocl/common/kernels_db.hpp"
#include "extension/implementation_params.hpp"
#include "extension/op_implementation.hpp"

namespace ov {
namespace ocl {

class BaseOCLImpl : public OpImplementation {
public:
    explicit BaseOCLImpl(std::string impl_name) : OpImplementation(impl_name, Type::OCL) {
        for (auto& name : m_db.get_names()) {
            std::cerr << "Found kernel: " << name << std::endl;
        }
    }


    virtual void init_kernel_data(const ImplementationParameters* params) = 0;
    KernelData get_kernel_data() const { return m_kernel_data; }

    void initialize(const ImplementationParameters* params) override;

protected:
    KernelData m_kernel_data;
    const ImplementationParameters* m_params;
    static const KernelsDataBase m_db;
};

}  // namespace ocl
}  // namespace ov
