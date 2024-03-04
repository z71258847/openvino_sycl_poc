// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extension/implementation_params.hpp"
#include "extension/implementation_registry.hpp"
#include "opset/fully_connected.hpp"

namespace ov {

struct FullyConnectedParams : public ImplementationParameters {
    FullyConnectedParams(const intel_gpu::op::FullyConnected* node) : ImplementationParameters(node) {}
};

class FullyConnectedImplementationsRegistry : public ImplementationsRegistry {
public:
    FullyConnectedImplementationsRegistry();
    static const FullyConnectedImplementationsRegistry& instance() {
        static FullyConnectedImplementationsRegistry instance;
        return instance;
    }
};

}  // namespace ov
