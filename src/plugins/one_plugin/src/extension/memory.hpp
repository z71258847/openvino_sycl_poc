// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "extension/memory_descriptor.hpp"

namespace ov {

class Memory {
public:
    virtual ~Memory() = default;

private:
    MemoryDesc m_desc;
};


}  // namespace ov
