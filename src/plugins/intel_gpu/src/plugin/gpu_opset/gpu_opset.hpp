// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node_extension.hpp"
#include "openvino/core/type.hpp"

#define DECLARE_GPU_OP(NewOpType, OriginalOpType) \
    class NewOpType : public OriginalOpType, public ov::intel_gpu::TypedNodeExtension<OriginalOpType, TypedNodeParams<OriginalOpType>> { \
    public: \
        using TypedNode = ov::intel_gpu::TypedNodeExtension<OriginalOpType, TypedNodeParams<OriginalOpType>>; \
        explicit NewOpType(std::shared_ptr<OriginalOpType> op) : OriginalOpType(*op) { \
            TypedNode::set_node_ptr(this); \
        } \
    }; \
    extern void __register_ ## NewOpType ## _factory() { \
        OpConverter::instance().register_converter(OriginalOpType::get_type_info_static(), \
        [](const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::Node> { \
            return std::make_shared<NewOpType>(std::dynamic_pointer_cast<OriginalOpType>(node)); \
        }); \
    }

namespace ov {
namespace intel_gpu {

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

}  // namespace intel_gpu
}  // namespace ov
