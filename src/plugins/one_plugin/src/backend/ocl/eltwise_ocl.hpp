// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base_ocl_impl.hpp"
#include "extension/implementation_params.hpp"

namespace ov {
namespace ocl {

class SomeEltwiseOCLImpl : public BaseOCLImpl {
public:
    SomeEltwiseOCLImpl() : BaseOCLImpl("SomeEltwiseOCLImpl") {}

    OpExecutor::Ptr get_executor(const ImplementationParameters* params) const override;
    bool supports(const ImplementationParameters* params) const override;
};

}  // namespace ocl
}  // namespace ov
