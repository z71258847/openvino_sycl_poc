// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>
#include "implementation_registry.hpp"
#include "implementation_params.hpp"


namespace ov {
namespace intel_gpu {

class ImplementationsFactory {
public:
    ImplementationsFactory(const ImplementationsList& impls) : m_impls(impls) {}
    std::shared_ptr<OpImplementation> create_impl(const FactoryParameters& params);
    void filter_unsupported(const FactoryParameters& params);

    virtual bool supports(const FactoryParameters& params) const = 0;

protected:
    ImplementationsList m_impls;
};

template <typename NodeType, typename ParametersType, typename RegistryType,
          typename std::enable_if<std::is_base_of<FactoryParameters, ParametersType>::value &&
                                  std::is_base_of<ov::Node, NodeType>::value &&
                                  std::is_base_of<ImplementationsRegistry, RegistryType>::value, bool>::type = true>
class TypedImplementationsFactory : public ImplementationsFactory {
public:
    TypedImplementationsFactory() : ImplementationsFactory(RegistryType::instance().get_all_impls()) {
        std::cerr << "Create impls factory for " << NodeType::get_type_info_static().name << std::endl;
        for (auto& impl : m_impls)
            std::cerr << impl->get_implementation_name() << std::endl;
    }

    bool supports(const FactoryParameters& params) const override {
        return supports_impl(static_cast<const ParametersType&>(params));
    };

protected:
    virtual bool supports_impl(const ParametersType& params) const { return false; };
};

}  // namespace op
}  // namespace ov
