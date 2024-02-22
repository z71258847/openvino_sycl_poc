// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "joint_impl/op_implementation.hpp"

namespace ov {

struct ImplSelector {
    static std::shared_ptr<ImplSelector> default_cpu_selector();
    static std::shared_ptr<ImplSelector> default_gpu_selector();

    virtual ImplementationBuilder select_best_implementation(const BuildersList& list) {
        return list.front();
    }
};

struct GPUImplSelector : public ImplSelector {
    ImplementationBuilder select_best_implementation(const BuildersList& list) override;
};

struct CPUImplSelector : public ImplSelector {
    ImplementationBuilder select_best_implementation(const BuildersList& list) override;
};

}  // namespace ov
