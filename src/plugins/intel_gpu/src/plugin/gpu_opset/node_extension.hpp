// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gpu_opset/implementation_factory.hpp"
#include "gpu_opset/implementation_args.hpp"
#include "gpu_opset/implementation_params.hpp"
#include "gpu_opset/memory_descriptor.hpp"
#include "gpu_opset/optimization_attributes.hpp"

#include <memory>


namespace ov {
namespace intel_gpu {

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

    const ov::Node* get_node_ptr() const;
    void set_node_ptr(const ov::Node* ptr);

protected:
    MemoryDescs m_memory_desc;
    std::shared_ptr<ImplementationsFactory> m_factory;
    std::shared_ptr<OptimizationAttributes> m_opt_attributes = nullptr;
    std::shared_ptr<ov::Model> m_fused_ops = nullptr;
    const ov::Node* m_node;
};

template <typename NodeType, typename std::enable_if<std::is_base_of<ov::Node, NodeType>::value, bool>::type = true>
class TypedNodeExtensionBase : public NodeExtension {
public:
    template<typename FactoryType, typename std::enable_if<std::is_base_of<ImplementationsFactory, FactoryType>::value, bool>::type = true>
    void init_factory() {
        m_factory = std::make_shared<FactoryType>();
    }
    template<typename FactoryType, typename std::enable_if<std::is_base_of<ImplementationsFactory, FactoryType>::value, bool>::type = true>
    FactoryType& get_factory() const {
        return static_cast<FactoryType&>(m_factory);
    }
};

template <typename NodeType, typename std::enable_if<std::is_base_of<ov::Node, NodeType>::value, bool>::type = true>
class TypedNodeExtension : public TypedNodeExtensionBase<NodeType> { };

}  // namespace op
}  // namespace ov
