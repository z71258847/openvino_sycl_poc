// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "implementation_selector.hpp"
#include "joint_impl/op_implementation.hpp"

namespace ov {

std::shared_ptr<ImplSelector> ImplSelector::default_cpu_selector() {
    return std::make_shared<CPUImplSelector>();
}

std::shared_ptr<ImplSelector> ImplSelector::default_gpu_selector() {
    return std::make_shared<GPUImplSelector>();
}

OpImplementation::Ptr GPUImplSelector::select_best_implementation(const ImplementationsList& list, const ov::Node* node) {
    return list.back();
}

OpImplementation::Ptr CPUImplSelector::select_best_implementation(const ImplementationsList& list, const ov::Node* node) {
    return list.front();
}

}  // namespace ov
