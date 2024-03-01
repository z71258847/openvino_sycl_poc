// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base_ocl_impl.hpp"
#include "extension/implementation_params.hpp"
#include "extension/executor.hpp"

namespace ov {
namespace ocl {

class SomeEltwiseOCLImpl : public BaseOCLImpl {
public:
    SomeEltwiseOCLImpl() : BaseOCLImpl("SomeEltwiseOCLImpl") {}

    OpExecutor::Ptr get_executor() const override;
    bool supports(const ImplementationParameters* params) const override;
    void init_kernel_data(const ImplementationParameters* params) override;
};

}  // namespace ocl
}  // namespace ov
