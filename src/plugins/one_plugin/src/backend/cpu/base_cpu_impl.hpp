// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "extension/implementation_params.hpp"
#include "extension/op_implementation.hpp"

namespace ov {
namespace cpu {

class BaseCPUImpl : public OpImplementation {
public:
    explicit BaseCPUImpl(std::string impl_name) : OpImplementation(impl_name, Type::CPU) { }

    void initialize(const ImplementationParameters* params) override;
protected:
    const ImplementationParameters* m_params;
};

}  // namespace cpu
}  // namespace ov
