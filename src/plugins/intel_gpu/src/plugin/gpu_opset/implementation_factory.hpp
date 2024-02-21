// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>
#include "gpu_opset/op_implementation.hpp"
#include "implementation_registry.hpp"
#include "implementation_params.hpp"
#include "openvino/core/node.hpp"


namespace ov {
namespace intel_gpu {

class ImplementationsFactory {
public:
    ImplementationsFactory(const ImplementationsList& impls) : m_impls(impls) {}
    virtual std::shared_ptr<OpImplementation> create_impl(const ov::Node* node) { return nullptr; }
    // ImplementationsList filter_unsupported(const FactoryParameters& params, const ImplementationsList& all_impls) {
    //     ImplementationsList res;
    //     for (auto impl : m_impls) {
    //         // TODO: implement filtering logic
    //         res.push_back(impl);
    //         break;
    //     }

    //     return res;
    // }

    virtual bool supports(const FactoryParameters& params) const = 0;

protected:
    std::shared_ptr<OpImplementation> m_preferred_impl = nullptr;
    ImplementationsList m_impls;
};

template <typename NodeType, typename ParametersType, typename RegistryType,
          typename std::enable_if<std::is_base_of<FactoryParameters, ParametersType>::value &&
                                  std::is_base_of<ov::Node, NodeType>::value &&
                                  std::is_base_of<ImplementationsRegistry, RegistryType>::value, bool>::type = true>
class TypedImplementationsFactory : public ImplementationsFactory {
public:
    TypedImplementationsFactory()
        : ImplementationsFactory(RegistryType::instance().get_all_impls()) {
        std::cerr << "Create impls factory for " << NodeType::get_type_info_static().name << std::endl;
        for (auto& impl : m_impls)
            std::cerr << impl->get_implementation_name() << std::endl;
    }

    bool supports(const FactoryParameters& params) const override {
        return supports_impl(static_cast<const ParametersType&>(params));
    };

    std::shared_ptr<OpImplementation> create_impl(const ov::Node* node) override {
        auto params = ParametersType(static_cast<const NodeType*>(node));
        OPENVINO_ASSERT(!m_impls.empty());
        m_preferred_impl = m_impls.front();
        return m_preferred_impl;
    }

protected:
    bool supports_impl(const ParametersType& params) const { return false; };
};

}  // namespace op
}  // namespace ov
