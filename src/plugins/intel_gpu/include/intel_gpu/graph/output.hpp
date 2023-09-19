// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"

namespace ov {
namespace intel_gpu {

// Output class represents an output port of primitive
// It has similar API as `memory` with few extension
// The objects are supposed to be created once for each primitive regardless actual shapes type
class Output {
public:
    explicit Output(cldnn::engine& engine);
    explicit Output(cldnn::memory::ptr memory);

    void set_memory(cldnn::memory::ptr memory);
    cldnn::memory::ptr get_memory() const;

    void allocate(const cldnn::layout& layout, cldnn::allocation_type type, bool reset = true);
    void deallocate();
    bool allocated() const;

    bool can_reinterpret(const cldnn::layout& new_layout);
    void reinterpret(const cldnn::layout& new_layout);

    cldnn::allocation_type get_allocation_type() const;
    const cldnn::layout& get_layout() const;
    size_t get_actual_size() const;
    size_t size() const;

    void set_reused(bool flag);

    void set_external(bool flag);
    bool is_external() const;

    cldnn::event::ptr fill(cldnn::stream& stream);

private:
    cldnn::engine& m_engine;
    cldnn::memory::ptr m_memory = nullptr;
    size_t actual_elements_count = 0;
    bool external = false;
};

}  // namespace intel_gpu
}  // namespace ov

namespace cldnn {
using ov::intel_gpu::Output;
}  // namespace cldnn
