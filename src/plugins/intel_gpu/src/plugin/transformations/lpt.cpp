// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt.hpp"

#include <memory>
#include <vector>

#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/low_precision.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/recurrent_cell.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"

namespace ov {
namespace intel_gpu {

bool LowPrecisionTransformations::run_on_model(const std::shared_ptr<ov::Model>& model) {
    using namespace ov::pass::low_precision;

    using const_node_ptr = const std::shared_ptr<const ov::Node>;
    const auto& default_precisions = ov::pass::low_precision::precision_set::get_int8_support();

    auto supportedPrecisions = std::vector<PrecisionsRestriction>(
        {PrecisionsRestriction::create<ov::op::v1::Convolution>({
             {{0}, {ov::element::u8, ov::element::i8}},
             {{1}, {ov::element::i8}},
         }),
         PrecisionsRestriction::create<ov::op::v1::ConvolutionBackpropData>(
             {{{0}, {ov::element::u8, ov::element::i8}}, {{1}, {ov::element::i8}}}),
         PrecisionsRestriction::create<ov::op::v1::GroupConvolution>({{{0}, {ov::element::u8, ov::element::i8}}, {{1}, {ov::element::i8}}}),
         PrecisionsRestriction::create<ov::op::v5::LSTMSequence>(PrecisionsRestriction::PrecisionsByPorts{}),
         PrecisionsRestriction::create<ov::op::v5::GRUSequence>(PrecisionsRestriction::PrecisionsByPorts{})});

    auto perTensorQuantization = std::vector<QuantizationGranularityRestriction>({
        QuantizationGranularityRestriction::create<ov::op::v1::Convolution>({0}),
        QuantizationGranularityRestriction::create<ov::op::v1::ConvolutionBackpropData>({0}),
    });

    ov::pass::Manager manager;

    auto pass_config = manager.get_pass_config();
    // quantized LSTMSequence / GPUSequence are not supported yet. Avoid extra transformation
    pass_config->disable<ov::pass::low_precision::RecurrentCellTransformation>();
    pass_config->set_callback<ov::pass::low_precision::MarkupPrecisions>([](const_node_ptr& node) -> bool {
        if (const auto mulitply = std::dynamic_pointer_cast<const ov::op::v1::Multiply>(node)) {
            return !MultiplyToGroupConvolutionTransformation::canBeTransformedToGroupConvolution(mulitply);
        }
        return false;
    });
    pass_config->set_callback<ConvolutionBackpropDataTransformation>([default_precisions](const_node_ptr& node) -> bool {
        auto fillStaticChannel = [](const ov::PartialShape& shape, size_t& channel) -> bool {
            const auto rank = shape.rank();
            if (rank.is_dynamic()) {
                return false;
            }
            if (rank.get_length() < 2l) {
                return false;
            }
            const auto& dimension = shape[1];
            if (dimension.is_dynamic()) {
                return false;
            }
            channel = dimension.get_length();
            return true;
        };

        size_t inputChannels = 0;
        if (!fillStaticChannel(node->get_input_partial_shape(0), inputChannels)) {
            return true;
        }

        size_t outputChannels = 0;
        if (!fillStaticChannel(node->get_output_partial_shape(0), outputChannels)) {
            return true;
        }

        if ((inputChannels % 4 != 0) || (outputChannels % 16 != 0)) {
            return true;
        }

        return LayerTransformation::isAsymmetricQuantization(node, default_precisions) ||
               WeightableLayerTransformation::isAsymmetricOnWeights(node, default_precisions);
    });

    pass_config->set_callback<MultiplyToGroupConvolutionTransformation>([&](const_node_ptr& node) -> bool {
        // disable MultiplyToGroupConvolution if Multiply with Constant can be fused

        const auto dequantization = NetworkHelper::getDequantization(node, default_precisions, 0, true);
        std::shared_ptr<ov::Node> parent = dequantization.empty() ? nullptr : dequantization.data.get_node()->shared_from_this();
        if (parent == nullptr) {
            const auto constantNode = NetworkHelper::getConstantInput(node);
            const auto constant = constantNode == nullptr ? nullptr : ov::as_type_ptr<ov::op::v0::Constant>(constantNode);
            if (constant != nullptr) {
                auto parent = node->get_input_node_shared_ptr(0);
                if (parent == constant) {
                    parent = node->get_input_node_shared_ptr(1);
                }
            }
        }

        if (parent != nullptr) {
            const auto parentHasOneConsumer = parent->get_output_target_inputs(0).size() == 1ul;
            if (parentHasOneConsumer) {
                return true;
            }
        }

        // disable MultiplyToGroupConvolution for Multiply with scalar

        if (MultiplyToGroupConvolutionTransformation::isDynamicOrScalar(node)) {
            return true;
        }

        return false;
    });

    pass_config->set_callback<ov::pass::ConvertQuantizeDequantize>([&](const_node_ptr& node) -> bool {
        return ov::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node, default_precisions);
    });

    bool reshapeIgnorePerTensorQuantizationCheck = false;
    if (m_context.has_dpas())  // Disable reshape transform until onednn i8 fc is optimized
        reshapeIgnorePerTensorQuantizationCheck = true;
    auto params = LayerTransformation::Params(true, element::f32, default_precisions, reshapeIgnorePerTensorQuantizationCheck);
    manager.register_pass<ov::pass::ConvertQuantizeDequantize>();
    manager.register_pass<LowPrecision>(supportedPrecisions, perTensorQuantization, params);
    return manager.run_passes(model);
}
}  // namespace intel_gpu
}  // namespace ov
