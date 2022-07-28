// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/tile.hpp"

#include "intel_gpu/primitives/tile.hpp"

namespace ov {
namespace intel_gpu {

static void CreateTileOp(Program& p, const std::shared_ptr<ngraph::op::v0::Tile>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    size_t rank = op->get_input_shape(0).size();

    auto repeats_node = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
    if (!repeats_node)
        IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() <<
                                                        " (" << op->get_type_name() << ")";
    std::vector<int64_t> repeats = repeats_node->cast_vector<int64_t>();

    int64_t default_size = 1;
    for (size_t i = repeats.size(); i < rank; ++i) {
        repeats.insert(repeats.begin(), default_size);
    }
    auto tilePrim = cldnn::tile(layerName,
                                inputPrimitives[0],
                                repeats,
                                op->get_friendly_name());

    p.AddPrimitive(tilePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, Tile);

}  // namespace intel_gpu
}  // namespace ov
