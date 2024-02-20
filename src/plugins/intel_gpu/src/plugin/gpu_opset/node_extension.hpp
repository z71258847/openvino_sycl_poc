// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/op.hpp"

#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/runtime/format.hpp"
#include "openvino/op/parameter.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "intel_gpu/graph/primitive_impl.hpp"

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace ov {
namespace intel_gpu {

using Format = cldnn::format;
using ImplTypes = cldnn::impl_types;

class PrimitiveImplementation {
public:
    virtual void execute() = 0;
};

class SomeNodeImpl : public PrimitiveImplementation {
    void execute() override {
        std::cerr << "SomeNodeImpl::execute()!\n";
    }
};

using ImplementationsList = std::vector<std::shared_ptr<PrimitiveImplementation>>;

class ImplementationsRegistry {
public:
    static ImplementationsRegistry& instance() {
        static ImplementationsRegistry instance;
        return instance;
    }

    ImplementationsList get_all_impls() const { return m_impls; }

private:
    ImplementationsRegistry() {
        register_impl<SomeNodeImpl>();
    }
    template <typename ImplType, typename std::enable_if<std::is_base_of<PrimitiveImplementation, ImplType>::value, bool>::type = true>
    void register_impl() {
        m_impls.push_back(std::make_shared<ImplType>());
    }

    ImplementationsList m_impls;
};

struct FactoryParameters {

};

template <typename NodeType>
struct TypedNodeParams : FactoryParameters {
    std::string some_parameter = "";
};

template<>
struct TypedNodeParams<ov::op::v0::Parameter> : public FactoryParameters {
    int some_parameter = 0;
};

template<>
struct TypedNodeParams<ov::op::v0::MatMul> : public FactoryParameters {
    bool some_parameter = false;
};

class ImplementationsFactory {
public:
    ImplementationsFactory(const ImplementationsList& impls) : m_impls(impls) {}
    std::shared_ptr<PrimitiveImplementation> create_impl(const FactoryParameters& params);
    void filter(const FactoryParameters& params);

    virtual bool supports(const FactoryParameters& params) const = 0;

private:
    ImplementationsList m_impls;
};

template <typename NodeType, typename ParametersType>
class TypedImplementationsFactory : public ImplementationsFactory {
public:
    TypedImplementationsFactory() : ImplementationsFactory(ImplementationsRegistry::instance().get_all_impls()) {}

    std::shared_ptr<PrimitiveImplementation> create(const FactoryParameters& attr);
    void filter(const FactoryParameters& attr);

    bool supports(const FactoryParameters& params) const override {
        return supports_impl(static_cast<const ParametersType&>(params));
    };

protected:
    virtual bool supports_impl(const ParametersType& params) = 0;

private:
    ImplementationsRegistry m_registry;
};

struct OptimizationAttributes {
    bool m_inplace;
};

struct MemoryDesc {
    MemoryDesc()
        : m_format(Format::any)
        , m_data_type(ov::element::undefined)
        , m_shape(ov::PartialShape::dynamic())
        , m_pad_b(ov::PartialShape::dynamic())
        , m_pad_e(ov::PartialShape::dynamic()) {}

    Format m_format;
    element::Type m_data_type;
    ov::PartialShape m_shape;
    ov::PartialShape m_pad_b; // need partialshape here ?
    ov::PartialShape m_pad_e; // need partialshape here ?
};

struct Argument {
    static Argument input(size_t id) {
        OPENVINO_ASSERT(id < max_inputs_size);
        return Argument(inputs_offset + id);
    }

    static Argument output(size_t id) {
        OPENVINO_ASSERT(id < max_outputs_size);
        return Argument(outputs_offset + id);
    }

    static Argument weights() {
        return Argument(weights_offset);
    }

    static Argument bias() {
        return Argument(bias_offset);
    }

    static Argument post_op(size_t id) {
        return Argument(post_op_offset);
    }

    operator size_t() const { return m_arg_id; }

    Argument(const Argument& other) = default;
    Argument(Argument&& other) = default;
    Argument& operator=(const Argument& other) = default;
    Argument& operator=(Argument&& other) = default;

private:
    size_t m_arg_id;
    Argument(size_t id) : m_arg_id(id) {}

    static constexpr const size_t max_inputs_size = 32;
    static constexpr const size_t max_outputs_size = 32;
    static constexpr const size_t max_weights_size = 1;
    static constexpr const size_t max_bias_size = 1;

    static constexpr const size_t inputs_offset = 0;
    static constexpr const size_t outputs_offset = max_inputs_size;
    static constexpr const size_t weights_offset = max_inputs_size + max_outputs_size;
    static constexpr const size_t bias_offset = max_inputs_size + max_outputs_size + weights_offset;
    static constexpr const size_t post_op_offset = max_inputs_size + max_outputs_size + weights_offset + max_bias_size;
};

class NodeExtension {
public:
    using MemoryDescs = std::map<Argument, MemoryDesc>;
    virtual ~NodeExtension() = default;

    NodeExtension() = default;
    NodeExtension(const element::TypeVector& _input_data_types, const element::TypeVector& _output_data_types) {}

    void visit_attributes(AttributeVisitor& visitor) {}

    const MemoryDescs& get_memory_desc() const { return m_memory_desc; }
    void set_memory_desc(const Argument& arg, const MemoryDesc& desc) { m_memory_desc[arg] = desc; }
    void set_memory_descs(const MemoryDescs& descs) { m_memory_desc = descs; }

    void set_inplace() { m_opt_attributes->m_inplace = true; }
    bool is_inplace() const { return m_opt_attributes->m_inplace; }

protected:
    MemoryDescs m_memory_desc;
    std::shared_ptr<ImplementationsFactory> m_factory;
    std::shared_ptr<OptimizationAttributes> m_opt_attributes = nullptr;
    std::shared_ptr<ov::Model> m_fused_ops = nullptr;
};

template <typename NodeType, typename ParametersType>
class TypedNodeExtension : public NodeExtension {
public:
    using FactoryType = TypedImplementationsFactory<NodeType, ParametersType>;
    ~TypedNodeExtension() = default;

    FactoryType& get_factory() const {
        return static_cast<FactoryType&>(m_factory);
    }
};

}  // namespace op
}  // namespace ov
