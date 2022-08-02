// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/plugin/program.hpp>
#include <intel_gpu/plugin/common_utils.hpp>

#include <intel_gpu/primitives/range.hpp>
#include <ngraph/op/range.hpp>

namespace ov {
namespace intel_gpu {

static void CreateRangeOp(Program &p, const std::shared_ptr<ngraph::op::v4::Range> &op) {
    p.ValidateInputs(op, { 3 });
    auto& out_pshape = op->get_output_partial_shape(0);

    auto outDataType = DataTypeFromPrecision(op->get_output_element_type(0));
    cldnn::layout outLayout { out_pshape, outDataType, cldnn::format::bfyx };
    cldnn::range prim { layer_type_name_ID(op), p.GetInputPrimitiveIDs(op), outLayout, op->get_friendly_name() };
    p.AddPrimitive(prim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v4, Range);

}  // namespace intel_gpu
}  // namespace ov
