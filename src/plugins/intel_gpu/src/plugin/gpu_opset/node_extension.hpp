// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gpu_opset/implementation_factory.hpp"
#include "gpu_opset/implementation_args.hpp"
#include "gpu_opset/implementation_params.hpp"
#include "gpu_opset/memory_descriptor.hpp"
#include "gpu_opset/optimization_attributes.hpp"
#include "intel_gpu/primitives/implementation_desc.hpp"

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

    void select_preferred_formats();

    const ov::Node* get_node_ptr() const;
    void set_node_ptr(const ov::Node* ptr);

protected:
    MemoryDescs m_memory_desc;
    std::shared_ptr<ImplementationsFactory> m_factory;
    std::shared_ptr<OptimizationAttributes> m_opt_attributes = nullptr;
    std::shared_ptr<ov::Model> m_fused_ops = nullptr;
    const ov::Node* m_node;
};

template <typename NodeType, typename ParametersType>
class TypedNodeExtension : public NodeExtension {
public:
    using FactoryType = TypedImplementationsFactory<NodeType, ParametersType>;
    TypedNodeExtension() {
        m_factory = std::make_shared<FactoryType>();
    }
    ~TypedNodeExtension() = default;

    FactoryType& get_factory() const {
        return static_cast<FactoryType&>(m_factory);
    }
};

}  // namespace op
}  // namespace ov
