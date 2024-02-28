// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend/cpu/base_cpu_impl.hpp"
#include "extension/implementation_params.hpp"

namespace ov {
namespace cpu {

class SomeBatchToSpaceCPUImpl : public BaseCPUImpl {
public:
    SomeBatchToSpaceCPUImpl() : BaseCPUImpl("SomeBatchToSpaceCPUImpl") {}

    OpExecutor::Ptr get_executor() const override;
    bool supports(const ImplementationParameters* params) const override;
};

}  // namespace cpu
}  // namespace ov
