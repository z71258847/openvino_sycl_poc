// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "joint_impl/extended_opset.hpp"
#include "joint_impl/implementation_params.hpp"
#include "joint_impl/implementation_registry.hpp"
#include "intel_gpu/op/fully_connected.hpp"

namespace ov {

class SomeFCImpl : public OpImplementation {
public:
    SomeFCImpl(const FactoryParameters& params) : OpImplementation("SomeFCImpl") {}

    void execute() override {
        std::cerr << "SomeFCImpl::execute()!\n";
    }
};


class FullyConnectedImplementationsRegistry : public ImplementationsRegistry<FactoryParameters> {
public:
    FullyConnectedImplementationsRegistry() {
        register_impl<SomeFCImpl>();
    }
    static const FullyConnectedImplementationsRegistry& instance() {
        static FullyConnectedImplementationsRegistry instance;
        return instance;
    }
};

REGISTER_OP_1(FullyConnected, ov::intel_gpu::op::FullyConnected, FactoryParameters, FullyConnectedImplementationsRegistry);

}  // namespace ov
