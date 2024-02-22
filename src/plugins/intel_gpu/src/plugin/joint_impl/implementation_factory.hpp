// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "joint_impl/op_implementation.hpp"
#include "implementation_params.hpp"
#include "implementation_selector.hpp"
#include "openvino/core/node.hpp"


namespace ov {

class ImplementationsFactory {
public:
    virtual ~ImplementationsFactory() = default;
    virtual std::shared_ptr<OpImplementation> create_impl(const ov::Node* node) = 0;

    std::shared_ptr<FactoryParameters> m_params = nullptr;
    std::shared_ptr<ImplSelector> m_impl_selector = nullptr;


    BuildersList m_available_impls;
};

template<typename NodeType, typename ParametersType>
class TypedFactory : public ImplementationsFactory{
public:
    ParametersType get_params() { return *std::dynamic_pointer_cast<ParametersType>(m_params); }

    explicit TypedFactory(const ov::Node* node) : TypedFactory(dynamic_cast<const NodeType*>(node)) { }
    explicit TypedFactory(const NodeType* node) {
        m_params = make_params(node);
        m_impl_selector = ImplSelector::default_gpu_selector(); // can be parameterized with affinity/requested device/some other params
    }

    std::shared_ptr<OpImplementation> create_impl(const ov::Node* node) override {
        OPENVINO_ASSERT(m_params != nullptr);
        return m_available_impls.front()(*m_params);
    }

protected:
    std::shared_ptr<ParametersType> make_params(const NodeType* node) const {
        return std::make_shared<ParametersType>(node);
    }

    BuildersList filter_unsupported(const ParametersType& params, const BuildersList& impls) const {
        BuildersList supported;
        for (const auto& impl : impls) {
            supported.push_back(impl);
        }

        return supported;
    }


};


}  // namespace ov
