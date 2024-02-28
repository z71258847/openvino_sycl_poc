// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extension/implementation_builder.hpp"
#include "backend/ocl/impl_builder.hpp"

namespace ov {

ImplementationBuilders::ImplementationBuilders() {
        m_builders = {
        { OpImplementation::Type::OCL, std::make_shared<ocl::OCLImplementationBuilder>() },
        { OpImplementation::Type::CPU, std::make_shared<CPUImplementationBuilder>() },
        { OpImplementation::Type::UNKNOWN, std::make_shared<CPUImplementationBuilder>() }
    };
}
}  // namespace ov
