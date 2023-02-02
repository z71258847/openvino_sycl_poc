// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"

#include <memory>

namespace cldnn {
struct primitive_impl;
}  // namespace cldnn

namespace ov {
namespace intel_gpu {

struct KernelsCache {
public:
    virtual ~KernelsCache() = default;
    virtual void compile_sequential() = 0;
    virtual void compile_parallel(InferenceEngine::CPUStreamsExecutor::Ptr task_executor = nullptr) = 0;
    virtual void reset() = 0;
    void init_primitive_impl(cldnn::primitive_impl& impl);
    static std::unique_ptr<KernelsCache> create(cldnn::engine& engine,
                                                const ExecutionConfig& config,
                                                uint32_t prog_id);

    virtual void save(cldnn::BinaryOutputBuffer& ob) const = 0;
    virtual void load(cldnn::BinaryInputBuffer& ib) = 0;
};
}  // namespace intel_gpu
}  // namespace ov

namespace cldnn {
using ov::intel_gpu::KernelsCache;
}
