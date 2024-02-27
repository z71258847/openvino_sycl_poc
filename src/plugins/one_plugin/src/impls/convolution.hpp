// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "extension/implementation_params.hpp"
#include "extension/implementation_registry.hpp"
#include "opset/convolution.hpp"

namespace ov {

struct SomeCustomParams : ImplementationParameters {
    SomeCustomParams(const ov::intel_gpu::op::Convolution* node) : ImplementationParameters(node) {}
};

class ConvolutionImplementationsRegistry : public ImplementationsRegistry {
public:
    ConvolutionImplementationsRegistry();
    static const ConvolutionImplementationsRegistry& instance() {
        static ConvolutionImplementationsRegistry instance;
        return instance;
    }
};

}  // namespace ov
