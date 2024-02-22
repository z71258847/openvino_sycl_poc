// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "implementation_selector.hpp"

namespace ov {

std::shared_ptr<ImplSelector> ImplSelector::default_cpu_selector() {
    return std::make_shared<CPUImplSelector>();
}

std::shared_ptr<ImplSelector> ImplSelector::default_gpu_selector() {
    return std::make_shared<GPUImplSelector>();
}

}  // namespace ov
