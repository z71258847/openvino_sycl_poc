// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gpu_op_extension.hpp"
#include "openvino/core/type.hpp"
#include "openvino/opsets/opset12.hpp"
#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/gather_compressed.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/rms.hpp"
#include "intel_gpu/op/reorder.hpp"

#define DECLARE_GPU_OP(NewOpType, OriginalOpType) \
    class NewOpType : public OriginalOpType, public ov::intel_gpu::GPUOpExtension { \
    public: \
        explicit NewOpType(std::shared_ptr<OriginalOpType> op) : OriginalOpType(*op) {} \
    };


namespace ov {
namespace intel_gpu {

class OpConverter {
public:
    void register_converter(ov::DiscreteTypeInfo source_type, std::function<std::shared_ptr<ov::Node>(const std::shared_ptr<ov::Node>&)> f);
    std::shared_ptr<ov::Node> convert_to_gpu_opset(const std::shared_ptr<ov::Node>& op) const;

    void register_ops();

private:
    std::unordered_map<ov::DiscreteTypeInfo, std::function<std::shared_ptr<ov::Node>(const std::shared_ptr<ov::Node>&)>> m_conversion_map;
};

const OpConverter& gpu_op_converter();

template<typename T, typename... Args>
std::shared_ptr<ov::Node> make_gpu_op(Args... args) {
    auto common_op = std::make_shared<T>(std::forward<Args>(args)...);
    auto gpu_op = intel_gpu::gpu_op_converter().convert_to_gpu_opset(common_op);
    gpu_op->set_output_size(common_op->get_output_size());
    gpu_op->set_friendly_name(common_op->get_friendly_name());
    gpu_op->validate_and_infer_types();

    return gpu_op;
}

#define _OPENVINO_OP_REG(NewOpType, OriginalOpType) DECLARE_GPU_OP(NewOpType, OriginalOpType)
#include "gpu_opset_tbl.hpp"
#undef _OPENVINO_OP_REG

}  // namespace intel_gpu
}  // namespace ov
