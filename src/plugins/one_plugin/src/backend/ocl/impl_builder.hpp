// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "base_ocl_impl.hpp"
#include "common/kernel_data.hpp"
#include "extension/implementation_builder.hpp"

namespace ov {
namespace ocl {

struct OCLImplementationBuilder : public ImplementationBuilder {
    using Ptr = std::shared_ptr<OCLImplementationBuilder>;

    void build() override {
        std::cerr << "Compile ALL ocl kernels!\n";
    }

    void add_impl(OpImplementation::Ptr impl) override {
        if (auto ocl_impl = std::dynamic_pointer_cast<BaseOCLImpl>(impl)) {
            kernels_data.push_back(ocl_impl->get_kernel_data());
        }
    }

    std::vector<KernelData> kernels_data;
};

}  // namespace ocl
}  // namespace ov
