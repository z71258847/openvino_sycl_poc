// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/reshape.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/op/unsqueeze.hpp"

#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/reorder.hpp"

namespace ov {
namespace intel_gpu {

static void CreateCommonReshapeOp(Program& p, const std::shared_ptr<ngraph::Node>& op) {
    p.ValidateInputs(op, {1, 2});
    auto input_primitives = p.GetInputPrimitiveIDs(op);
    std::string layer_name = layer_type_name_ID(op);

    auto input_pshape = op->get_input_partial_shape(0);
    auto output_pshape = op->get_output_partial_shape(0);

    OPENVINO_ASSERT(input_pshape.is_static() && output_pshape.is_static(), "Dynamic shapes are not supported for Reshape operation yet");

    auto outTensor = tensor_from_dims(output_pshape.to_shape());

    // if we convert from or to 5D/6D, additional reorder also required to change format
    cldnn::primitive_id reshapeInputId = input_primitives[0];
    if (input_pshape.size() != output_pshape.size()) {
        cldnn::primitive_id reorderId = "reorder:" + op->get_friendly_name() + "_reorder";
        cldnn::format outputFormat = cldnn::format::bfyx;

        switch (output_pshape.size()) {
        case 5: outputFormat = cldnn::format::bfzyx; break;
        case 6: outputFormat = cldnn::format::bfwzyx; break;
        default: break;
        }

        cldnn::layout outputLayout(DataTypeFromPrecision(op->get_output_element_type(0)), outputFormat, outTensor);
        p.AddPrimitive(cldnn::reorder(reorderId,
                                      reshapeInputId,
                                      outputLayout,
                                      std::vector<float>(),
                                      cldnn::reorder_mean_mode::subtract,
                                      op->get_friendly_name()));
        p.InitProfileInfo(reorderId, "Reorder", false, InferenceEngine::InferenceEngineProfileInfo::EXECUTED, layer_name);
        p.primitiveIDs[layer_name + "_reorder"] = reorderId;
        p.primitiveIDs[reorderId] = reorderId;
        p.profilingIDs.push_back(reorderId);
        reshapeInputId = reorderId;
    }

    auto reshapePrim = cldnn::reshape(layer_name,
                                      reshapeInputId,
                                      outTensor,
                                      op->get_friendly_name());

    p.AddPrimitive(reshapePrim);
    p.AddPrimitiveToProfiler(op);
}

static void CreateReshapeOp(Program& p, const std::shared_ptr<ngraph::op::v1::Reshape>& op) {
    p.ValidateInputs(op, {1, 2});
    auto input_primitives = p.GetInputPrimitiveIDs(op);
    std::string layer_name = layer_type_name_ID(op);

    auto out_pshape = op->get_output_partial_shape(0);
    auto pattern_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));

    if (pattern_constant) {
        std::vector<int64_t> shape_pattern = pattern_constant->cast_vector<int64_t>();
        auto reshapePrim = cldnn::reshape(layer_name,
                                          input_primitives[0],
                                          op->get_special_zero(),
                                          shape_pattern,
                                          out_pshape,
                                          op->get_friendly_name());

        p.AddPrimitive(reshapePrim);
    } else {
        auto shape_prim_id = input_primitives[1];
        auto reshapePrim = cldnn::reshape(layer_name,
                                          input_primitives[0],
                                          shape_prim_id,
                                          op->get_special_zero(),
                                          out_pshape,
                                          op->get_friendly_name());

        p.AddPrimitive(reshapePrim);
    }
    p.AddPrimitiveToProfiler(op);
}

static void CreateSqueezeOp(Program& p, const std::shared_ptr<ngraph::op::v0::Squeeze>& op) {
    CreateCommonReshapeOp(p, op);
}

static void CreateUnsqueezeOp(Program& p, const std::shared_ptr<ngraph::op::v0::Unsqueeze>& op) {
    CreateCommonReshapeOp(p, op);
}

REGISTER_FACTORY_IMPL(v1, Reshape);
REGISTER_FACTORY_IMPL(v0, Squeeze);
REGISTER_FACTORY_IMPL(v0, Unsqueeze);

}  // namespace intel_gpu
}  // namespace ov
