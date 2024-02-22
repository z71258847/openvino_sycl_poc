// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

#include "joint_impl/implementation_factory.hpp"
#include "joint_impl/implementation_args.hpp"
#include "joint_impl/implementation_params.hpp"
#include "joint_impl/memory_descriptor.hpp"
#include "joint_impl/op_implementation.hpp"
#include "joint_impl/optimization_attributes.hpp"


#include <memory>


namespace ov {

class NodeExtension {
public:
    virtual ~NodeExtension() = default;

    NodeExtension() = default;

    void visit_attributes(AttributeVisitor& visitor);

    const MemoryDescs& get_memory_desc() const;
    void set_memory_desc(const Argument& arg, const MemoryDesc& desc);
    void set_memory_descs(const MemoryDescs& descs);

    void set_inplace();
    bool is_inplace() const;

    virtual void select_preferred_formats();
    virtual void select_best_implementation() = 0;
    std::shared_ptr<OpImplementation> get_impl() const;

    const ov::Node* get_node_ptr() const;
    void set_node_ptr(const ov::Node* ptr);

protected:
    MemoryDescs m_memory_desc;
    std::shared_ptr<ImplementationsFactory> m_factory;
    std::shared_ptr<OpImplementation> m_best_implementation = nullptr;
    std::shared_ptr<OptimizationAttributes> m_opt_attributes = nullptr;
    std::shared_ptr<ov::Model> m_fused_ops = nullptr;
    const ov::Node* m_node;
};

template <typename NodeType,
          typename std::enable_if<std::is_base_of<ov::Node, NodeType>::value, bool>::type = true>
class TypedNodeExtensionBase : public NodeExtension {
public:
    template<typename FactoryType, typename std::enable_if<std::is_base_of<ImplementationsFactory, FactoryType>::value, bool>::type = true>
    void init(const ov::Node* ptr) {
        set_node_ptr(ptr);
        m_factory = std::make_shared<FactoryType>(ptr);
    }
    template<typename FactoryType, typename std::enable_if<std::is_base_of<ImplementationsFactory, FactoryType>::value, bool>::type = true>
    FactoryType& get_factory() const {
        return static_cast<FactoryType&>(m_factory);
    }

    void select_best_implementation() override {
        m_best_implementation = m_factory->create_impl(m_node);
    }
};

template <typename NodeType, typename std::enable_if<std::is_base_of<ov::Node, NodeType>::value, bool>::type = true>
class TypedNodeExtension : public TypedNodeExtensionBase<NodeType> { };

}  // namespace ov
