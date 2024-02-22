// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "joint_impl/extended_opset.hpp"
#include "joint_impl/implementation_params.hpp"
#include "joint_impl/implementation_registry.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include <memory>

namespace ov {

using NodeType = ov::intel_gpu::op::FullyConnected;
using ParametersType = TypedNodeParams<NodeType>;

class SomeFCImpl : public OpImplementation {
public:
    SomeFCImpl(const ParametersType& params) : OpImplementation("SomeFCImpl") {}

    void execute() override {
        std::cerr << "SomeFCImpl::execute()!\n";
    }
};


class FullyConnectedImplementationsRegistry : public ImplementationsRegistry<ParametersType> {
public:
    FullyConnectedImplementationsRegistry() {
        register_impl<SomeFCImpl>();
    }
    static const FullyConnectedImplementationsRegistry& instance() {
        static FullyConnectedImplementationsRegistry instance;
        return instance;
    }
};

REGISTER_OP_1(FullyConnected, ov::intel_gpu::op::FullyConnected, TypedNodeParams<NodeType>, FullyConnectedImplementationsRegistry);

}  // namespace ov
