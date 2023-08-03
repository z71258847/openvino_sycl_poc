// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/memory_caps.hpp"
#include "memory.hpp"
#include "engine.hpp"

namespace ov {
namespace intel_gpu {

class MemoryManager {
public:
    using Ptr = std::shared_ptr<MemoryManager>;

    explicit MemoryManager(cldnn::engine& engine) : m_engine(engine) {}
    explicit MemoryManager(cldnn::memory::ptr memory) : m_engine(*memory->get_engine()), m_memory(memory) {}

    void set_memory(cldnn::memory::ptr memory) {
        m_memory = memory;
        actual_elements_count = m_memory->count();
    }

    void allocate(const cldnn::layout& layout, cldnn::allocation_type type, bool reset = true) {
        m_memory = m_engine.allocate_memory(layout, type, reset);
        actual_elements_count = m_memory->count();
    }

    bool allocated() const {
        return m_memory != nullptr;
    }

    void deallocate() {
        m_memory.reset();
        actual_elements_count = 0;
    }

    size_t get_actual_size() const { return actual_elements_count; }

    bool can_reinterpret(const cldnn::layout& new_layout) {
        return allocated() && actual_elements_count >= new_layout.count();
    }

    void reinterpret(const cldnn::layout& new_layout) {
        OPENVINO_ASSERT(can_reinterpret(new_layout));
        m_memory = m_engine.reinterpret_buffer(*m_memory, new_layout);
    }

    const cldnn::layout& get_layout() const {
        return m_memory->get_layout();
    }

    cldnn::memory::ptr get_memory() const {
        return m_memory;
    }

    cldnn::allocation_type get_allocation_type() const {
        return m_memory->get_allocation_type();
    }
    size_t size() const {
        return m_memory->size();
    }

    void set_reused(bool flag) {
        m_memory->set_reused(flag);
    }

    cldnn::event::ptr fill(cldnn::stream& stream) {
        return m_memory->fill(stream);
    }

private:
    cldnn::engine& m_engine;
    cldnn::memory::ptr m_memory = nullptr;
    size_t actual_elements_count = 0;
};

}  // namespace intel_gpu
}  // namespace ov

namespace cldnn {
using ov::intel_gpu::MemoryManager;
}  // namespace cldnn
