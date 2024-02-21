// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu_opset.hpp"

#include "gpu_opset/implementation_registry.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset12.hpp"
#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/gather_compressed.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/rms.hpp"
#include "intel_gpu/op/reorder.hpp"
#include "intel_gpu/op/convolution.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/op/indirect_gemm.hpp"
#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/op/swiglu.hpp"
#include "intel_gpu/op/swiglu.hpp"
#include "ov_ops/multiclass_nms_ie_internal.hpp"
#include "ov_ops/nms_static_shape_ie.hpp"
#include "ov_ops/nms_ie_internal.hpp"
#include "ov_ops/generate_proposals_ie_internal.hpp"


namespace ov {
namespace intel_gpu {

void OpConverter::register_converter(ov::DiscreteTypeInfo source_type, std::function<std::shared_ptr<ov::Node>(const std::shared_ptr<ov::Node>&)> f) {
    m_conversion_map[source_type] = f;
}

std::shared_ptr<ov::Node> OpConverter::convert_to_gpu_opset(const std::shared_ptr<ov::Node>& op) const {
    OPENVINO_ASSERT(m_conversion_map.count(op->get_type_info()) > 0, "[GPU] Operation ", op->get_type_info(), " is not registered");
    auto converted_op = m_conversion_map.at(op->get_type_info())(op);
    converted_op->set_output_size(op->get_output_size());
    converted_op->set_friendly_name(op->get_friendly_name());
    ov::copy_runtime_info(op, converted_op);
    return converted_op;
}

void OpConverter::register_ops() {
#define REGISTER_FACTORY(NewOpType, OpType) extern void __register_ ## NewOpType ## _factory(); __register_ ## NewOpType ## _factory();
#include "gpu_opset_tbl.hpp"
REGISTER_FACTORY(Abs_v0, ov::op::v0::Abs);
REGISTER_FACTORY(Relu_v0, ov::op::v0::Relu);
REGISTER_FACTORY(BatchToSpace_v1, ov::op::v1::BatchToSpace);
REGISTER_FACTORY(Convolution_internal, ov::intel_gpu::op::Convolution);
REGISTER_FACTORY(FullyConnectedCompressed_internal, ov::intel_gpu::op::FullyConnectedCompressed);
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

#define REGISTER_FACTORY(NewOpType, OpType) REGISTER_OP(NewOpType, OpType, RegistryStub)
#include "gpu_opset_tbl.hpp"
#undef REGISTER_FACTORY

}  // namespace intel_gpu
}  // namespace ov
