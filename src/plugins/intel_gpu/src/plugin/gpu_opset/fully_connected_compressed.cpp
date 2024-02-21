// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu_opset.hpp"
#include "gpu_opset/implementation_params.hpp"
#include "gpu_opset/implementation_registry.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"

namespace ov {
namespace intel_gpu {


using NodeType = ov::intel_gpu::op::FullyConnectedCompressed;
using ParametersType = TypedNodeParams<NodeType>;

class FullyConnectedImplementationsRegistry : public ImplementationsRegistry {
public:
    FullyConnectedImplementationsRegistry() {
        // register_impl<>()
    }
    static const FullyConnectedImplementationsRegistry& instance() {
        static FullyConnectedImplementationsRegistry instance;
        return instance;
    }
};

template <>
class TypedImplementationsFactory<NodeType, ParametersType, FullyConnectedImplementationsRegistry> : public ImplementationsFactory {
public:
    TypedImplementationsFactory() : ImplementationsFactory(FullyConnectedImplementationsRegistry::instance().get_all_impls()) {
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

REGISTER_OP(FullyConnectedCompressed_internal, ov::intel_gpu::op::FullyConnectedCompressed, FullyConnectedImplementationsRegistry);

}  // namespace intel_gpu
}  // namespace ov
