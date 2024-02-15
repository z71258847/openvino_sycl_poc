// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_transformations.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "convert_pooling_to_reduce.hpp"
#include "convert_shapeof.hpp"
#include "decompose_reduce_for_false_keepdims.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/lstm_cell_fusion.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp"
#include "transformations/common_optimizations/wrap_interpolate_into_transposes.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_batch_to_space.hpp"
#include "transformations/op_conversions/convert_broadcast3.hpp"
#include "transformations/op_conversions/convert_depth_to_space.hpp"
#include "transformations/op_conversions/convert_gather_0d.hpp"
#include "transformations/op_conversions/convert_gather_downgrade.hpp"
#include "transformations/op_conversions/convert_gelu.hpp"
#include "transformations/op_conversions/convert_gp9_to_gp_ie_internal.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"
#include "transformations/op_conversions/convert_matrix_nms_to_matrix_nms_ie.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
#include "transformations/op_conversions/convert_multiclass_nms_to_multiclass_nms_ie.hpp"
#include "transformations/op_conversions/convert_nms9_to_nms_ie_internal.hpp"
#include "transformations/op_conversions/convert_nms_rotated_to_nms_ie_internal.hpp"
#include "transformations/op_conversions/convert_pad12_downgrade.hpp"
#include "transformations/op_conversions/convert_previous_nms_to_nms_9.hpp"
#include "transformations/op_conversions/convert_prior_box_v8_to_v0.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/convert_reduce_to_reshape.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "transformations/op_conversions/convert_shapeof3.hpp"
#include "transformations/op_conversions/convert_shuffle_channels3.hpp"
#include "transformations/op_conversions/convert_softmax_downgrade.hpp"
#include "transformations/op_conversions/convert_space_to_batch.hpp"
#include "transformations/op_conversions/convert_space_to_depth.hpp"
#include "transformations/op_conversions/convert_ti_to_sequences.hpp"
#include "transformations/op_conversions/convert_topk11_downgrade.hpp"
#include "transformations/op_conversions/eye_decomposition.hpp"
#include "transformations/op_conversions/gelu7_downgrade.hpp"
#include "transformations/op_conversions/gru_cell_decomposition.hpp"
#include "transformations/op_conversions/hsigmoid_decomposition.hpp"
#include "transformations/op_conversions/hswish_decomposition.hpp"
#include "transformations/op_conversions/log_softmax_decomposition.hpp"
#include "transformations/op_conversions/lstm_cell_decomposition.hpp"
#include "transformations/op_conversions/mvn6_decomposition.hpp"
#include "transformations/op_conversions/normalize_l2_decomposition.hpp"
#include "transformations/op_conversions/reduce_l1_decomposition.hpp"
#include "transformations/op_conversions/reduce_l2_decomposition.hpp"
#include "transformations/op_conversions/rnn_cell_decomposition.hpp"
#include "transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp"
#include "transformations/op_conversions/softmax_decomposition.hpp"
#include "transformations/op_conversions/softplus_decomposition.hpp"
#include "transformations/opset_conversions/convert_opset2_to_opset1.hpp"
#include "transformations/opset_conversions/convert_opset3_to_opset2.hpp"
#include "transformations/smart_reshape/matmul_sr.hpp"
#include "transformations/utils/utils.hpp"

namespace {
template <typename T>
static bool disable_reduce_decomposition(const std::shared_ptr<const ov::Node> node) {
    if (auto op = std::dynamic_pointer_cast<const T>(node)) {
        if (op->input(0).get_partial_shape()[0].is_static()) {
            bool fp16_batch_not_1 = op->get_element_type() == ov::element::f16 && op->input(0).get_partial_shape()[0] != 1;
            return !fp16_batch_not_1;
        }
    }
    return false;
}

}  // namespace

namespace ov {
namespace intel_gpu {

bool CommonTransformations::run_on_model(const std::shared_ptr<ov::Model>& model) {
    using const_node_ptr = const std::shared_ptr<const ov::Node>;

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    auto pass_config = manager.get_pass_config();
    manager.register_pass<ov::pass::CommonOptimizations>();

    manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
    manager.register_pass<ov::pass::TransposeSinking>();

    if (!m_context.unroll_loop()) {
        manager.register_pass<ov::pass::BidirectionalLSTMSequenceDecomposition>();
        manager.register_pass<ov::pass::BidirectionalGRUSequenceDecomposition>();
        manager.register_pass<ov::pass::BidirectionalRNNSequenceDecomposition>();
    }

    manager.register_pass<ov::pass::ConvertSequenceToTensorIterator>();
    manager.register_pass<ov::pass::ConvertOpSet3ToOpSet2>();
    manager.register_pass<ov::pass::ConvertOpSet2ToOpSet1>();

    manager.register_pass<ov::pass::LSTMCellDecomposition>();
    manager.register_pass<ov::pass::GRUCellDecomposition>();
    manager.register_pass<ov::pass::RNNCellDecomposition>();

    if (m_context.unroll_loop()) {
        manager.register_pass<ov::pass::BidirectionalLSTMSequenceDecomposition>();
        manager.register_pass<ov::pass::BidirectionalGRUSequenceDecomposition>();
        manager.register_pass<ov::pass::BidirectionalRNNSequenceDecomposition>();
    }

    manager.register_pass<ConvertShapeOf1To3>();
    manager.register_pass<ov::pass::ConvertNMS1ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS3ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS4ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS5ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS9ToNMSIEInternal>();
    manager.register_pass<ov::pass::ConvertNMSRotatedToNMSIEInternal>();
    manager.register_pass<ov::pass::ConvertGP9ToGPIEInternal>();
    manager.register_pass<ov::pass::ConvertMatrixNmsToMatrixNmsIE>();
    manager.register_pass<ov::pass::ConvertGather0D>();
    manager.register_pass<ov::pass::ConvertPriorBox8To0, false>();
    manager.register_pass<ov::pass::ConvertMulticlassNmsToMulticlassNmsIE>();
    manager.register_pass<ov::pass::TransposeMatMul>();
    manager.register_pass<ov::pass::ConvertPad12ToPad1, false>();
    manager.register_pass<ov::pass::Validate>();

    precisions_map int_convert_precision_map{
        {ov::element::i64, ov::element::i32},
        {ov::element::u64, ov::element::i32},
        {ov::element::u16, ov::element::i32},
        {ov::element::u32, ov::element::i32},
        {ov::element::boolean, ov::element::u8},
        {ov::element::i4, ov::element::i8},
        {ov::element::u4, ov::element::u8},
    };

    type_to_fuse_map empty_fuse_map = {};
    const bool keep_precision_sensitive_in_fp32 = true;
    const bool convert_input_output_precision = false;
    manager.register_pass<ov::pass::ConvertPrecision>(int_convert_precision_map,
                                                      empty_fuse_map,
                                                      keep_precision_sensitive_in_fp32,
                                                      convert_input_output_precision);

    pass_config->disable<ov::pass::EyeDecomposition>();

    // disable conversion to legacy and use the new mixed precision
    // in which precision sensitive nodes are kept in FP32
    pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();

    // SpaceToDepth/DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
    pass_config->set_callback<ov::pass::ConvertSpaceToDepth, ov::pass::ConvertDepthToSpace>([](const_node_ptr& node) -> bool {
        return node->input_value(0).get_partial_shape().size() <= 5lu &&
               node->input_value(0).get_partial_shape().size() == node->get_output_partial_shape(0).size();
    });

    pass_config->set_callback<ov::pass::ConvertBatchToSpace, ov::pass::ConvertSpaceToBatch>([](const_node_ptr& node) -> bool {
        const auto& rank = node->input(0).get_partial_shape().rank().get_length();
        return rank <= 5;
    });

    // Convert reduce to reshape expected to be optimized out
    manager.register_pass<ov::pass::ConvertReduceToReshape>();

    if (m_context.has_dpas()) {
        // oneDNN reduction is used
        pass_config->disable<ov::pass::ConvertReduceSumToPooling>();
        pass_config->disable<ov::pass::ConvertReduceMeanToPooling>();
        pass_config->disable<ov::pass::ConvertReduceMaxToPooling>();
        manager.register_pass<ConvertAvgPoolingToReduce>();
        manager.register_pass<DecomposeReduceForFalseKeepDims>();
    } else {
        pass_config->set_callback<ov::pass::ConvertReduceSumToPooling>([](const_node_ptr& node) -> bool {
            return disable_reduce_decomposition<ov::op::v1::ReduceSum>(node);
        });

        pass_config->set_callback<ov::pass::ConvertReduceMeanToPooling>([](const_node_ptr& node) -> bool {
            return disable_reduce_decomposition<ov::op::v1::ReduceMean>(node);
        });

        pass_config->set_callback<ov::pass::ConvertReduceMaxToPooling>([](const_node_ptr& node) -> bool {
            return disable_reduce_decomposition<ov::op::v1::ReduceMax>(node);
        });
    }

    auto isCellPrimitiveSupported = [](const_node_ptr& node) -> bool {
        if (std::dynamic_pointer_cast<const ov::op::v0::RNNCell>(node)) {
            return false;
        } else if (std::dynamic_pointer_cast<const ov::op::v3::GRUCell>(node)) {
            return false;
        } else if (const auto& lstm_cell = std::dynamic_pointer_cast<const ov::op::v4::LSTMCell>(node)) {
            return lstm_cell->get_clip() == 0.0f && lstm_cell->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
        } else if (const auto& lstm_cell_v1 = std::dynamic_pointer_cast<const ov::op::v0::LSTMCell>(node)) {
            return lstm_cell_v1->get_clip() == 0.0f &&
                   lstm_cell_v1->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
        }
        return false;
    };

    // Sequences supported by the plugin shouldn't be converted to TensorIterator.
    // sequence_length input is not supported in all Sequences, so if is_seq_len_provided() == true, we
    // should always convert to TensorIterator.
    // RNN/GRU Sequences are not supported in GPU plugin
    // LSTM Sequence supported with clip == 0, and activations have default values (sigmoid, tanh, tanh)
    auto isSequencePrimitiveSupported = [](const_node_ptr& node) -> bool {
        const auto& data = node->input(0);
        const auto& data_pshape = data.get_partial_shape();
        if (data_pshape.rank().is_static() && data_pshape.rank().get_length() > 1 && !data_pshape[1].is_static())
            return false;
        auto max_seq_len = data.get_shape().at(1);
        if (std::dynamic_pointer_cast<const ov::op::v5::RNNSequence>(node)) {
            return false;
        } else if (std::dynamic_pointer_cast<const ov::op::v5::GRUSequence>(node)) {
            return false;
        } else if (const auto& lstm_seq = std::dynamic_pointer_cast<const ov::op::v5::LSTMSequence>(node)) {
            return lstm_seq->get_clip() == 0.0f && lstm_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"} &&
                   max_seq_len < 16 &&
                   !ov::op::util::is_seq_len_provided(lstm_seq->get_input_node_shared_ptr(0), lstm_seq->get_input_node_shared_ptr(3));
        }
        return false;
    };

    pass_config->set_callback<ov::pass::RNNCellDecomposition, ov::pass::GRUCellDecomposition, ov::pass::LSTMCellDecomposition>(
        [isCellPrimitiveSupported](const_node_ptr& node) -> bool {
            return isCellPrimitiveSupported(node);
        });

    pass_config->set_callback<ov::pass::LSTMCellFusion>([isCellPrimitiveSupported](const_node_ptr& node) -> bool {
        return !isCellPrimitiveSupported(node);
    });

    if (m_context.unroll_loop()) {
        pass_config->set_callback<ov::pass::ConvertRNNSequenceToTensorIterator,
                                  ov::pass::ConvertGRUSequenceToTensorIterator,
                                  ov::pass::ConvertLSTMSequenceToTensorIterator>(
            [isSequencePrimitiveSupported](const_node_ptr& node) -> bool {
                return isSequencePrimitiveSupported(node);
            });
    }

    pass_config->set_callback<ov::pass::ConvertLoopToLSTMSequence,
                              ov::pass::FuseReverseLSTMSequence,
                              ov::pass::FuseLSTMSequencesToBidirectionalLSTMSequence>(
        [isSequencePrimitiveSupported](const_node_ptr& node) -> bool {
            return !isSequencePrimitiveSupported(node);
        });

    pass_config->set_callback<ov::pass::MVN6Decomposition>([](const_node_ptr& node) -> bool {
        const auto mvn = std::dynamic_pointer_cast<const ov::op::v6::MVN>(node);
        if (mvn != nullptr && node->get_input_size() == 2) {
            if (auto axes_node = dynamic_cast<ov::op::v0::Constant*>(mvn->get_input_node_ptr(1))) {
                auto mvn_axes = axes_node->cast_vector<int64_t>();
                auto out_rank = mvn->get_output_partial_shape(0).size();
                ov::util::normalize_axes(mvn.get(), out_rank, mvn_axes);

                std::sort(mvn_axes.begin(), mvn_axes.end());

                // Supported cases:
                // 2 <= out_rank <= 5
                // axes set: [out_rank - 1, out_rank - 2, ... r] where r > 1
                // basically impl supports cases when tensor can be reshaped to [d1, d2]
                // so that d2 is set of dimensions for normalization

                // Skip unsupported ranks
                if (out_rank == 1 || out_rank > 5)
                    return false;

                // check axes set
                for (size_t i = 0; i < mvn_axes.size(); i++) {
                    auto axis = mvn_axes[mvn_axes.size() - i - 1];
                    if (axis != static_cast<int64_t>(out_rank - i - 1) || axis == 0) {
                        return false;
                    }
                }
                return true;
            }
        }
        return false;
    });

    pass_config->enable<ov::pass::NormalizeL2Decomposition>();
    pass_config->set_callback<ov::pass::NormalizeL2Decomposition>([](const_node_ptr& node) -> bool {
        // Condition to filter out axes such as [0, 1, 2] which is not supported currently.
        const auto norm = ov::as_type_ptr<const ov::op::v0::NormalizeL2>(node);
        const auto inputRank = norm->get_input_partial_shape(0).size();
        auto axesNode = ov::as_type_ptr<const ov::op::v0::Constant>(norm->get_input_node_shared_ptr(1));
        const auto axes = axesNode->cast_vector<size_t>();
        const auto isSupportedAxes = [](const std::vector<size_t>& axes, const size_t inputRank) {
            if (axes.size() == 1 && axes[0] == 1) {
                return true;
            } else if (axes.size() == inputRank - 1) {
                auto sortAxes = axes;
                std::sort(sortAxes.begin(), sortAxes.end());
                for (size_t i = 0; i < sortAxes.size(); i++) {
                    if (sortAxes[i] != i + 1)
                        return false;
                }
                return true;
            }
            return false;
        };

        if (!isSupportedAxes(axes, inputRank) && ov::shape_size(axesNode->get_shape()) != 0) {
            return false;
        }
        return true;
    });

    pass_config->enable<ov::pass::SoftmaxDecomposition>();
    pass_config->set_callback<ov::pass::SoftmaxDecomposition>([](const_node_ptr& node) -> bool {
        return node->input_value(0).get_partial_shape().rank().get_length() <= 5;
    });

    // List of enabled/disabled transformations
    pass_config->disable<ov::pass::ConvertGELU>();
    pass_config->disable<ov::pass::Gelu7Downgrade>();
    pass_config->disable<ov::pass::ConvertMod>();
    pass_config->disable<ov::pass::ConvertShuffleChannels3>();
    pass_config->disable<ov::pass::HSwishDecomposition>();
    pass_config->disable<ov::pass::HSigmoidDecomposition>();
    pass_config->disable<ov::pass::ReduceL1Decomposition>();
    pass_config->disable<ov::pass::ReduceL2Decomposition>();
    pass_config->disable<ov::pass::SoftPlusDecomposition>();
    pass_config->disable<ov::pass::LogSoftmaxDecomposition>();
    pass_config->disable<ov::pass::ConvertBroadcast3>();
    pass_config->disable<ov::pass::WeightsDequantizeToFakeQuantize>();
    pass_config->disable<ov::pass::SimplifyCTCGreedyDecoderSeqLen>();
    pass_config->disable<ov::pass::ConvertSoftMax8ToSoftMax1>();
    pass_config->disable<ov::pass::ConvertShapeOf3>();
    pass_config->disable<ov::pass::ConvertGather8ToGather7>();
    pass_config->disable<ov::pass::ConvertGather7ToGather1>();
    pass_config->disable<ov::pass::ConvertTopK11ToTopK3>();

    pass_config->enable<ov::pass::ConvertInterpolate1ToInterpolate4>();

    return manager.run_passes(model);
}
}  // namespace intel_gpu
}  // namespace ov
