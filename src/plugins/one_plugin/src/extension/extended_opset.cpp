// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extension/extended_opset.hpp"

#include "extension/implementation_params.hpp"
#include "extension/implementation_registry.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset12.hpp"
#include "opset/kv_cache.hpp"
#include "opset/read_value.hpp"
#include "opset/gather_compressed.hpp"
#include "opset/fully_connected.hpp"
#include "opset/fully_connected_compressed.hpp"
#include "opset/rms.hpp"
#include "opset/reorder.hpp"
#include "opset/convolution.hpp"
#include "opset/placeholder.hpp"
#include "opset/indirect_gemm.hpp"
#include "opset/gemm.hpp"
#include "opset/swiglu.hpp"
#include "opset/swiglu.hpp"
#include "ov_ops/multiclass_nms_ie_internal.hpp"
#include "ov_ops/nms_static_shape_ie.hpp"
#include "ov_ops/nms_ie_internal.hpp"
#include "ov_ops/generate_proposals_ie_internal.hpp"


namespace ov {

void OpConverter::register_converter(ov::DiscreteTypeInfo source_type, std::function<std::shared_ptr<ov::Node>(const std::shared_ptr<ov::Node>&)> f) {
    m_conversion_map[source_type] = f;
}

std::shared_ptr<ov::Node> OpConverter::convert_to_extended_opset(const std::shared_ptr<ov::Node>& op) const {
    OPENVINO_ASSERT(m_conversion_map.count(op->get_type_info()) > 0, "[GPU] Operation ", op->get_type_info(), " is not registered");
    auto converted_op = m_conversion_map.at(op->get_type_info())(op);
    converted_op->set_output_size(op->get_output_size());
    converted_op->set_friendly_name(op->get_friendly_name());
    ov::copy_runtime_info(op, converted_op);
    return converted_op;
}

void OpConverter::register_ops() {
#define REGISTER_FACTORY(NewOpType, OpType) \
    extern void __register_ ## NewOpType ## Extension ## _factory(); __register_ ## NewOpType ## Extension ## _factory();

#include "extended_opset_tbl.hpp"
REGISTER_FACTORY(Abs, ov::op::v0::Abs);
REGISTER_FACTORY(Relu, ov::op::v0::Relu);
REGISTER_FACTORY(BatchToSpace, ov::op::v1::BatchToSpace);
REGISTER_FACTORY(Convolution, ov::intel_gpu::op::Convolution);
REGISTER_FACTORY(Result, ov::op::v0::Result);
REGISTER_FACTORY(Parameter, ov::op::v0::Parameter);
REGISTER_FACTORY(Constant, ov::op::v0::Constant);
REGISTER_FACTORY(Placeholder, ov::intel_gpu::op::Placeholder);
// REGISTER_FACTORY(FullyConnectedCompressed_internal, ov::intel_gpu::op::FullyConnectedCompressed);
REGISTER_FACTORY(FullyConnected, ov::intel_gpu::op::FullyConnected);
REGISTER_FACTORY(Reshape, ov::op::v1::Reshape);
#undef REGISTER_FACTORY
}

OpConverter& OpConverter::instance() {
    static OpConverter op_converter;
    return op_converter;
}

class RegistryStub : public ImplementationsRegistry {
public:
    RegistryStub() { }
    static const RegistryStub& instance() {
        static RegistryStub instance;
        return instance;
    }
};

#define REGISTER_FACTORY(NewOpType, OpType) REGISTER_IMPLS(NewOpType, OpType, ImplementationParameters, RegistryStub)
#include "extended_opset_tbl.hpp"
#undef REGISTER_FACTORY

}  // namespace ov
