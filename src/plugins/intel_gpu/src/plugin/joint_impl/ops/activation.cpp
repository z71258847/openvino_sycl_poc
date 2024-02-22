// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "joint_impl/extended_opset.hpp"
#include "joint_impl/implementation_params.hpp"
#include "joint_impl/implementation_registry.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/relu.hpp"

namespace ov {

struct ActivationParams : public FactoryParameters {
    enum class Type {
        ReLU,
        Abs,
        Undef
    };
    Type type = Type::Undef;

    ActivationParams() = default;
    ActivationParams(const ov::op::v0::Abs* node) : type(Type::Abs) {}
    ActivationParams(const ov::op::v0::Relu* node) : type(Type::ReLU) {}
};

class SomeActivationImpl : public OpImplementation {
public:
    SomeActivationImpl(const ActivationParams& params) : OpImplementation("SomeActivationImpl"), m_params(params) {}

    void execute() override {
        std::cerr << "SomeActivationImpl::execute()!\n";
    }

    ActivationParams m_params;
};

struct ActivationImplementationsRegistry : public ImplementationsRegistry<ActivationParams> {
    ActivationImplementationsRegistry() {
        register_impl<SomeActivationImpl>();
    }
    static const ActivationImplementationsRegistry& instance() {
        static ActivationImplementationsRegistry instance;
        return instance;
    }
};


REGISTER_OP_1(Abs, ov::op::v0::Abs, ActivationParams, ActivationImplementationsRegistry);
REGISTER_OP_1(Relu, ov::op::v0::Relu, ActivationParams, ActivationImplementationsRegistry);

}  // namespace ov
