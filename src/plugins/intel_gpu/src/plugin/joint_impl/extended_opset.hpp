// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node_extension.hpp"
#include "implementation_params.hpp"
#include "openvino/core/type.hpp"


#define DECLARE_REGISTER_FUNC(NewOpType, OriginalOpType) \
    extern void __register_ ## NewOpType ## _factory(); \
    void __register_ ## NewOpType ## _factory() { \
        OpConverter::instance().register_converter(OriginalOpType::get_type_info_static(), \
        [](const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::Node> { \
            return std::make_shared<NewOpType>(std::dynamic_pointer_cast<OriginalOpType>(node)); \
        }); \
    }

#define DECLARE_NEW_OP_CLASS(NewOpType, OriginalOpType, FactoryType) \
    class NewOpType : public OriginalOpType, public TypedNodeExtension<OriginalOpType> { \
    public: \
        explicit NewOpType(std::shared_ptr<OriginalOpType> op) : OriginalOpType(*op) { \
            TypedNodeExtension<OriginalOpType>::init<FactoryType>(this); \
        } \
    } \

#define DECLARE_FACTORY_CLASS(FactoryName, OriginalOpType, TypedParams, Registry) \
    class FactoryName : public TypedFactory<OriginalOpType, TypedParams> { \
    public: \
        using Parent = TypedFactory<OriginalOpType, TypedParams>; \
        FactoryName(const ov::Node* node) : Parent(node) { \
            m_available_impls = filter_unsupported(Parent::get_params(), Registry::instance().all()); \
        } \
    }


#define REGISTER_OP_1(NewOpType, OriginalOpType, TypedParams, Registry) \
    DECLARE_FACTORY_CLASS(NewOpType ## Factory, OriginalOpType, TypedParams, Registry); \
    DECLARE_NEW_OP_CLASS(NewOpType ## Extension, OriginalOpType, NewOpType ## Factory); \
    DECLARE_REGISTER_FUNC(NewOpType ## Extension, OriginalOpType)


#define REGISTER_OP(NewOpType, OriginalOpType, RegistryType) \
    using FactoryType ## NewOpType = TypedImplementationsFactory<OriginalOpType, TypedNodeParams<OriginalOpType>, RegistryType>; \
    using TypedNode ## NewOpType = TypedNodeExtension<OriginalOpType>; \
    DECLARE_NEW_OP_CLASS(NewOpType, OriginalOpType, TypedNode ## NewOpType, FactoryType ## NewOpType) \
    DECLARE_REGISTER_FUNC(NewOpType, OriginalOpType)

#define REGISTER_OP_WITH_CUSTOM_FACTORY(NewOpType, OriginalOpType, FactoryType) \
    using TypedNode ## NewOpType = TypedNodeExtension<OriginalOpType>; \
    DECLARE_NEW_OP_CLASS(NewOpType, OriginalOpType, TypedNode ## NewOpType, FactoryType) \
    DECLARE_REGISTER_FUNC(NewOpType, OriginalOpType)

// Is it needed?
#define REGISTER_OP_WITH_CUSTOM_PARAMS_AND_REGISTRY(NewOpType, OriginalOpType, ParamsType, RegistryType) \
    using FactoryType ## NewOpType = TypedImplementationsFactory<OriginalOpType, ParamsType, RegistryType>; \
    using TypedNode ## NewOpType = TypedNodeExtension<OriginalOpType>; \
    DECLARE_NEW_OP_CLASS(NewOpType, OriginalOpType, TypedNode ## NewOpType, FactoryType ## NewOpType) \
    DECLARE_REGISTER_FUNC(NewOpType, OriginalOpType)



namespace ov {

class OpConverter {
public:
    using FactoryType = std::function<std::shared_ptr<ov::Node>(const std::shared_ptr<ov::Node>&)>;
    void register_converter(ov::DiscreteTypeInfo source_type, FactoryType f);
    std::shared_ptr<ov::Node> convert_to_gpu_opset(const std::shared_ptr<ov::Node>& op) const;
    static OpConverter& instance();
    void register_ops();

private:
    OpConverter() = default;
    std::unordered_map<ov::DiscreteTypeInfo, FactoryType> m_conversion_map;
};


template<typename T, typename... Args>
std::shared_ptr<ov::Node> make_gpu_op(Args... args) {
    auto common_op = std::make_shared<T>(std::forward<Args>(args)...);
    auto gpu_op = OpConverter::instance().convert_to_gpu_opset(common_op);
    gpu_op->set_output_size(common_op->get_output_size());
    gpu_op->set_friendly_name(common_op->get_friendly_name());
    gpu_op->validate_and_infer_types();

    return gpu_op;
}

}  // namespace ov
