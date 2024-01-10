// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu_opset.hpp"

#define REGISER_OP(NewOpType, OriginalOpType) \
    register_converter(OriginalOpType::get_type_info_static(), [](const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::Node> { \
        return std::make_shared<NewOpType>(std::dynamic_pointer_cast<OriginalOpType>(node)); \
    });

namespace ov {
namespace intel_gpu {

void OpConverter::register_converter(ov::DiscreteTypeInfo source_type, std::function<std::shared_ptr<ov::Node>(const std::shared_ptr<ov::Node>&)> f) {
    m_conversion_map[source_type] = f;
}

std::shared_ptr<ov::Node> OpConverter::convert_to_gpu_opset(const std::shared_ptr<ov::Node>& op) const {
    OPENVINO_ASSERT(m_conversion_map.count(op->get_type_info()) > 0, "[GPU] Operation ", op->get_type_info(), " is not registered");
    return m_conversion_map.at(op->get_type_info())(op);
}

void OpConverter::register_ops() {
#define _OPENVINO_OP_REG(NewOpType, OriginalOpType) REGISER_OP(NewOpType, OriginalOpType)
#include "gpu_opset_tbl.hpp"
#undef _OPENVINO_OP_REG
}

const OpConverter& gpu_op_converter() {
    static OpConverter op_converter;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
        op_converter.register_ops();
    });
    return op_converter;
}

}  // namespace intel_gpu
}  // namespace ov
