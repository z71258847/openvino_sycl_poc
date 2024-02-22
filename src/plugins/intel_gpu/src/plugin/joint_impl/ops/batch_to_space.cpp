// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_to_space.hpp"
#include "joint_impl/extended_opset.hpp"
#include "joint_impl/implementation_params.hpp"
#include "joint_impl/implementation_registry.hpp"
#include "openvino/op/batch_to_space.hpp"

namespace ov {

template<>
struct TypedNodeParams<ov::op::v1::BatchToSpace> : public FactoryParameters {
    TypedNodeParams() = default;
    explicit TypedNodeParams(const ov::op::v1::BatchToSpace* node) : some_parameter(node->get_output_size()) {

    }
    int some_parameter = 100500;
};
using BatchToSpaceParams = TypedNodeParams<ov::op::v1::BatchToSpace>;

class BatchToSpaceImpl : public OpImplementation {
public:
    BatchToSpaceImpl(const BatchToSpaceParams& params) : OpImplementation("BatchToSpaceImpl"), m_params(params) {}

    void execute() override {
        std::cerr << "BatchToSpaceImpl::execute(): " << m_params.some_parameter << std::endl;
    }

    BatchToSpaceParams m_params;
};

struct BatchToSpaceImplementationsRegistry : public ImplementationsRegistry<TypedNodeParams<ov::op::v1::BatchToSpace>> {
    BatchToSpaceImplementationsRegistry() {
        register_impl<BatchToSpaceImpl>();
    }
    static const BatchToSpaceImplementationsRegistry& instance() {
        static BatchToSpaceImplementationsRegistry instance;
        return instance;
    }
};

REGISTER_OP_1(BatchToSpace, ov::op::v1::BatchToSpace, TypedNodeParams<ov::op::v1::BatchToSpace>, BatchToSpaceImplementationsRegistry);

}  // namespace ov
