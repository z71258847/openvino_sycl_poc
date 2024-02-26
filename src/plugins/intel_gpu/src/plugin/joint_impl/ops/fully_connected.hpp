// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "joint_impl/implementation_params.hpp"
#include "joint_impl/implementation_registry.hpp"
#include "intel_gpu/op/fully_connected.hpp"

namespace ov {

struct FullyConnectedParams : public ImplementationParameters {
    FullyConnectedParams(const intel_gpu::op::FullyConnected* node) {}
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
