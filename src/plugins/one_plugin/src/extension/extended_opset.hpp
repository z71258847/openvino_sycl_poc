// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node_extension.hpp"
#include "implementation_params.hpp"
#include "openvino/core/type.hpp"


#define DECLARE_REGISTER_FUNC(NewOpType, OriginalOpType, FactoryType) \
    extern void __register_ ## NewOpType ## _factory(); \
    void __register_ ## NewOpType ## _factory() { \
        OpConverter::instance().register_converter(OriginalOpType::get_type_info_static(), \
        [](const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::Node> { \
            auto extended_op = std::make_shared<TypedNodeExtension<OriginalOpType>>(std::dynamic_pointer_cast<OriginalOpType>(node)); \
            extended_op->init_factory<FactoryType>(dynamic_cast<const ov::Node*>(extended_op.get())); \
            return extended_op; \
        }); \
    }

#define DECLARE_FACTORY_CLASS(FactoryName, OriginalOpType, TypedParams, Registry) \
    class FactoryName : public TypedFactory<OriginalOpType, TypedParams> { \
    public: \
        using Parent = TypedFactory<OriginalOpType, TypedParams>; \
        FactoryName(const ov::Node* node) : Parent(node, Registry::instance().all()) { } \
    }


#define REGISTER_IMPLS(NewOpType, OriginalOpType, TypedParams, Registry) \
    DECLARE_FACTORY_CLASS(NewOpType ## Factory, OriginalOpType, TypedParams, Registry); \
    DECLARE_REGISTER_FUNC(NewOpType ## Extension, OriginalOpType, NewOpType ## Factory)

namespace ov {

class OpConverter {
public:
    using FactoryType = std::function<std::shared_ptr<ov::Node>(const std::shared_ptr<ov::Node>&)>;
    void register_converter(ov::DiscreteTypeInfo source_type, FactoryType f);
    std::shared_ptr<ov::Node> convert_to_extended_opset(const std::shared_ptr<ov::Node>& op) const;
    static OpConverter& instance();
    void register_ops();

private:
    OpConverter() = default;
    std::unordered_map<ov::DiscreteTypeInfo, FactoryType> m_conversion_map;
};


template<typename T, typename... Args>
std::shared_ptr<ov::Node> make_gpu_op(Args... args) {
    auto common_op = std::make_shared<T>(std::forward<Args>(args)...);
    auto gpu_op = OpConverter::instance().convert_to_extended_opset(common_op);
    gpu_op->set_output_size(common_op->get_output_size());
    gpu_op->set_friendly_name(common_op->get_friendly_name());
    gpu_op->validate_and_infer_types();

    return gpu_op;
}

}  // namespace ov
