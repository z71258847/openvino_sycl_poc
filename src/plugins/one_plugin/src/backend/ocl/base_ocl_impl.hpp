// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "extension/implementation_params.hpp"
#include "extension/op_implementation.hpp"

namespace ov {
namespace ocl {

class BaseOCLImpl : public OpImplementation {
public:
    explicit BaseOCLImpl(std::string impl_name) : OpImplementation(impl_name, Type::OCL) {}
};

}  // namespace ocl
}  // namespace ov
