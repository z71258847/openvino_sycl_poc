// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/output.hpp"

namespace ov {
namespace intel_gpu {

Output::Output(cldnn::engine& engine)
    : m_engine(engine) {}

Output::Output(cldnn::memory::ptr memory)
    : m_engine(*memory->get_engine())
    , m_memory(memory)
    , actual_elements_count(memory ? m_memory->count() : 0) {}

void Output::set_memory(cldnn::memory::ptr memory) {
    m_memory = memory;
    actual_elements_count = memory ? m_memory->count() : 0;
}

cldnn::memory::ptr Output::get_memory() const {
    return m_memory;
}

void Output::allocate(const cldnn::layout& layout, cldnn::allocation_type type, bool reset) {
    m_memory = m_engine.allocate_memory(layout, type, reset);
    actual_elements_count = m_memory->count();
}

void Output::deallocate() {
    m_memory.reset();
    actual_elements_count = 0;
}

bool Output::allocated() const {
    return m_memory != nullptr;
}

bool Output::can_reinterpret(const cldnn::layout& new_layout) {
    return allocated() && actual_elements_count >= new_layout.count();
}

void Output::reinterpret(const cldnn::layout& new_layout) {
    OPENVINO_ASSERT(can_reinterpret(new_layout));
    m_memory = m_engine.reinterpret_buffer(*m_memory, new_layout);
}

cldnn::allocation_type Output::get_allocation_type() const {
    return m_memory->get_allocation_type();
}

const cldnn::layout& Output::get_layout() const {
    return m_memory->get_layout();
}

size_t Output::get_actual_size() const {
    return actual_elements_count;
}

size_t Output::size() const {
    return m_memory->size();
}

void Output::set_reused(bool flag) {
    m_memory->set_reused(flag);
}

void Output::set_external(bool flag) {
    external = flag;
}

bool Output::is_external() const {
    return external;
}

cldnn::event::ptr Output::fill(cldnn::stream& stream) {
    return m_memory->fill(stream);
}

}  // namespace intel_gpu
}  // namespace ov
