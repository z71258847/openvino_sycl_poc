// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_to_infer_precision.hpp"

#include <memory>
#include <vector>

#include "move_convert_after_gather.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/validate.hpp"
#include "transformations/common_optimizations/broadcast_elementwise_fusion.hpp"
#include "transformations/common_optimizations/broadcast_transition.hpp"
#include "transformations/common_optimizations/mvn_fusion.hpp"
#include "transformations/common_optimizations/softmax_fusion.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"
#include "transformations/op_conversions/mvn6_decomposition.hpp"

namespace {
static bool is_non_supported_decompression_op(const std::shared_ptr<const ov::Node> node) {
    auto get_single_consumer = [](const std::shared_ptr<const ov::Node> node) -> std::shared_ptr<ov::Node> {
        const auto consumers = node->get_output_target_inputs(0);
        if (consumers.size() != 1)
            return nullptr;
        return consumers.begin()->get_node()->shared_from_this();
    };

    auto consumer = get_single_consumer(node);
    if (!consumer)
        return true;

    if (ov::is_type<ov::op::v0::MatMul>(consumer) || ov::is_type<ov::op::v8::Gather>(consumer)) {
        return false;
    } else if (ov::is_type<ov::op::v1::Reshape>(consumer)) {
        consumer = get_single_consumer(consumer);
        if (consumer != nullptr && (ov::is_type<ov::op::v0::MatMul>(consumer) || ov::is_type<ov::op::v8::Gather>(consumer))) {
            return false;
        }
    }
    if (consumer != nullptr && ov::is_type<ov::op::v0::Convert>(consumer)) {
        consumer = get_single_consumer(consumer);
        if (consumer != nullptr && (ov::is_type<ov::op::v0::MatMul>(consumer) || ov::is_type<ov::op::v8::Gather>(consumer))) {
            return false;
        }
    }
    return true;
}
}  // namespace

namespace ov {
namespace intel_gpu {

bool ConvertToInferPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    using const_node_ptr = const std::shared_ptr<const ov::Node>;
    precisions_map fp_convert_precision_map = {{ov::element::f64, ov::element::f32}};

    const auto& device_info = m_context.get_device_info();
    const auto& config = m_context.get_config();
    // call conversion of float types with keep_precision_sensitive_in_fp32 = true
    auto fp_precision_supported = [&](ov::element::Type e) -> bool {
        switch (e) {
        case ov::element::f16:
            return device_info.supports_fp16;
        case ov::element::f32:
            return true;  // assume that all GPUs support f32 data type
        case ov::element::f64:
            return device_info.supports_fp64;
        case ov::element::bf16:
            return false;
        default:
            return false;
        }
        return false;
    };

    const auto fallback_precision = ov::element::f32;
    std::vector<ov::element::Type> fp_element_types = {ov::element::f32, ov::element::f16, ov::element::bf16};

    // Add conversion from FP data types to infer precision if it's specified
    auto infer_precision = config.get_property(ov::hint::inference_precision);
    if (infer_precision != ov::element::undefined) {
        if (!fp_precision_supported(infer_precision))
            infer_precision = fallback_precision;

        for (auto& et : fp_element_types) {
            if (et != infer_precision) {
                fp_convert_precision_map.insert({et, infer_precision});
            }
        }
    }

    // Add conversion from unsupported FP data types to f32 if we don't have a conversion to something valid already in the list
    for (auto& et : fp_element_types) {
        if (!fp_precision_supported(et)) {
            bool has_valid_conversion = fp_convert_precision_map.count(et) && fp_precision_supported(fp_convert_precision_map[et]);
            if (!has_valid_conversion) {
                fp_convert_precision_map.insert(std::make_pair(et, fallback_precision));
            }
        }
    }

    type_to_fuse_map empty_fuse_map = {};
    ov::pass::Manager manager;
    auto pass_config = manager.get_pass_config();
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::Validate>();

    // fuse softmax, MVN patterns, so that they will not be marked as precision sensitive in ConvertPrecision
    manager.register_pass<ov::pass::SoftmaxFusion>();
    manager.register_pass<ov::pass::MVNFusion>();
    // decompose MVNs that sre not supported in GPU, so that they will be marked as precision sensitive in ConvertPrecision
    manager.register_pass<ov::pass::MVN6Decomposition>();
    // Run these broadcast optimizations earlier to ensure that those are executed before NopElimination/ConstantFolding
    manager.register_pass<ov::pass::BroadcastElementwiseFusion>();
    manager.register_pass<ov::pass::BroadcastTransition>();

    manager.register_pass<ov::pass::KeepConstantsPrecisionAndAddConverts>();
    pass_config->set_callback<ov::pass::KeepConstantsPrecisionAndAddConverts>([](const_node_ptr& node) -> bool {
        auto next_node = node->get_output_target_inputs(0).begin()->get_node();
        if (is_type<ov::op::v0::Convert>(next_node)) {
            next_node = next_node->get_output_target_inputs(0).begin()->get_node();
        }
        return !is_type<ov::op::v0::MatMul>(next_node);
    });

    manager.register_pass<ov::pass::MarkDequantizationSubgraph>(ov::element::TypeVector{ov::element::u8, ov::element::u4, ov::element::i4},
                                                                true);
    // Need to check if transfomrations work correctly for mixed models with both compression and quantization at the same time.
    if (!m_context.is_model_quantized())
        pass_config->set_callback<ov::pass::MarkDequantizationSubgraph>(is_non_supported_decompression_op);

    manager.register_pass<ov::intel_gpu::MoveConvertAfterGather>();

    const bool keep_precision_sensitive_in_fp32_1 = true;
    const bool convert_input_output_precision = false;
    const bool store_original_precision_as_rt_attribute = true;
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map,
                                                      empty_fuse_map,
                                                      keep_precision_sensitive_in_fp32_1,
                                                      convert_input_output_precision,
                                                      store_original_precision_as_rt_attribute);

    return manager.run_passes(model);
}
}  // namespace intel_gpu
}  // namespace ov
