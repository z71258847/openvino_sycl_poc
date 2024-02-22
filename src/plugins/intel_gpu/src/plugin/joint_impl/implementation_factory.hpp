// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>
#include "joint_impl/op_implementation.hpp"
#include "implementation_registry.hpp"
#include "implementation_params.hpp"
#include "openvino/core/node.hpp"


namespace ov {

class ImplementationsFactory {
public:
    ImplementationsFactory(std::shared_ptr<FactoryParameters> params, const ImplementationsList& impls)
        : m_params(params)
        , m_impls(filter_unsupported(*params, impls)) {}
    virtual std::shared_ptr<OpImplementation> create_impl(const ov::Node* node) = 0;
    virtual bool supports(const FactoryParameters& params) const = 0;

protected:
    std::shared_ptr<FactoryParameters> m_params;
    std::shared_ptr<OpImplementation> m_preferred_impl = nullptr;
    ImplementationsList m_impls;

    ImplementationsList filter_unsupported(const FactoryParameters& params, const ImplementationsList& all_impls) {
        std::cerr << "Filter out unsupported impls base\n";
        ImplementationsList res;
        for (auto impl : all_impls) {
            // TODO: implement filtering logic
            res.push_back(impl);
            break;
        }

        return res;
    }
};

template <typename NodeType, typename ParametersType, typename RegistryType,
          typename std::enable_if<std::is_base_of<FactoryParameters, ParametersType>::value &&
                                  std::is_base_of<ov::Node, NodeType>::value &&
                                  std::is_base_of<ImplementationsRegistry, RegistryType>::value, bool>::type = true>
class TypedImplementationsFactory : public ImplementationsFactory {
public:
    TypedImplementationsFactory(const ov::Node* node)
        : ImplementationsFactory(std::make_shared<ParametersType>(dynamic_cast<const NodeType*>(node)), RegistryType::instance().get_all_impls()) {
        std::cerr << "Create impls factory for " << NodeType::get_type_info_static().name << std::endl;
        for (auto& impl : m_impls)
            std::cerr << impl->get_implementation_name() << std::endl;
    }

    bool supports(const FactoryParameters& params) const override {
        return supports_impl(static_cast<const ParametersType&>(params));
    };

    std::shared_ptr<OpImplementation> create_impl(const ov::Node* node) override {
        OPENVINO_ASSERT(!m_impls.empty(), "Can't create implementation for ", node->get_friendly_name(), " (type=", node->get_type_name(), ")");
        m_preferred_impl = m_impls.front();
        return m_preferred_impl;
    }

protected:
    bool supports_impl(const ParametersType& params) const { return false; };

};

}  // namespace ov
