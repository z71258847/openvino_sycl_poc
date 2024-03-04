// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory.hpp"


#include <vector>
#include <memory>
#include <stdexcept>

namespace ov {

Memory::Memory(Engine* engine, const MemoryDesc& desc)
    : m_engine(engine), m_desc(desc) {
    ptr = malloc(desc.byte_size());
}
Memory::Memory(Engine* engine, const MemoryDesc& desc, void* p)
    : ptr(p), external(true), m_engine(engine), m_desc(desc)  {
}

Memory::~Memory() {
    if (!external)
        free(ptr);
}

}  // namespace ov
