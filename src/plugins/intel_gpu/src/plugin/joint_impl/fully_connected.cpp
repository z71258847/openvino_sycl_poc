// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extended_opset.hpp"
#include "joint_impl/implementation_params.hpp"
#include "joint_impl/implementation_registry.hpp"
#include "intel_gpu/op/fully_connected.hpp"

namespace ov {

using NodeType = ov::intel_gpu::op::FullyConnected;
using ParametersType = TypedNodeParams<NodeType>;

class SomeFCImpl : public OpImplementation {
public:
    SomeFCImpl() : OpImplementation("SomeFCImpl") {}

    void initialize(const ParametersType& params) { }

    void execute() override {
        std::cerr << "SomeFCImpl::execute()!\n";
    }
};


class FullyConnectedImplementationsRegistry : public ImplementationsRegistry {
public:
    FullyConnectedImplementationsRegistry() {
        register_impl<SomeFCImpl>();
    }
    static const FullyConnectedImplementationsRegistry& instance() {
        static FullyConnectedImplementationsRegistry instance;
        return instance;
    }
};

template <>
class TypedImplementationsFactory<NodeType, ParametersType, FullyConnectedImplementationsRegistry, false> : public ImplementationsFactory {
public:
    TypedImplementationsFactory(const ov::Node* node)
        : ImplementationsFactory(std::make_shared<ParametersType>(dynamic_cast<const NodeType*>(node)),
                                 FullyConnectedImplementationsRegistry::instance().get_all_impls()) {
        std::cerr << "Specialized impls factory for " << NodeType::get_type_info_static().name << std::endl;
        for (auto& impl : m_impls)
            std::cerr << impl->get_implementation_name() << std::endl;
    }

    bool supports(const FactoryParameters& params) const override {
        return supports_impl(static_cast<const ParametersType&>(params));
    };

protected:
    virtual bool supports_impl(const ParametersType& params) const { return false; };
};

REGISTER_OP(FullyConnected_internal, ov::intel_gpu::op::FullyConnected, FullyConnectedImplementationsRegistry);

}  // namespace ov
