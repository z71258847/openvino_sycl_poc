// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "joint_impl/extended_opset.hpp"
#include "joint_impl/extended_opset.hpp"
#include "joint_impl/implementation_registry.hpp"
#include "joint_impl/implementation_params.hpp"
#include "joint_impl/op_implementation.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {

struct ElementwiseParams : public FactoryParameters {
    enum class Type {
        Add,
        Sub,
        Undef
    };
    Type type = Type::Undef;

    ElementwiseParams() = default;
    ElementwiseParams(const ov::op::v1::Add* node) : type(Type::Add) {}
    ElementwiseParams(const ov::op::v1::Subtract* node) : type(Type::Sub) {}
};

class SomeEltwiseImpl : public OpImplementation {
public:
    SomeEltwiseImpl(const ElementwiseParams& params) : OpImplementation("SomeEltwiseImpl"), m_params(params) {}

    void execute() override {
        std::cerr << "SomeEltwiseImpl::execute(): " << (int)m_params.type << std::endl;
    }

    const ElementwiseParams& m_params;
};

struct ElementwiseRegistry : public ImplementationsRegistry<ElementwiseParams> {
    ElementwiseRegistry() {
        register_impl<SomeEltwiseImpl>();
    }
    static const ElementwiseRegistry& instance() {
        static ElementwiseRegistry instance;
        return instance;
    }
};

REGISTER_OP_1(Add, op::v1::Add, ElementwiseParams, ElementwiseRegistry);
REGISTER_OP_1(Subtract, op::v1::Subtract, ElementwiseParams, ElementwiseRegistry);

}  // namespace ov
