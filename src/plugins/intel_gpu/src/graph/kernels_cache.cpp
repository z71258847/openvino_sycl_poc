// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/kernels_cache.hpp"
#include "primitive_inst.h"
#include "implementation_map.hpp"

namespace ov {
namespace intel_gpu {
void KernelsCache::init_primitive_impl(cldnn::primitive_impl& impl) {
    impl.add_to_cache(*this);
    compile_sequential();
    impl.init_kernels(*this);
    reset();
}

std::unique_ptr<KernelsCache> KernelsCache::create(cldnn::engine& engine,
                                                   const ExecutionConfig& config,
                                                   uint32_t prog_id) {
    auto factory = cldnn::KernelsCacheFactory::get(cldnn::impl_types::ocl);
    return factory(engine, config, prog_id);
}
}  // namespace intel_gpu
}  // namespace ov
