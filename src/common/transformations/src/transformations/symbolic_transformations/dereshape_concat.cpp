// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/dereshape_concat.hpp"
#include <memory>

#include "itt.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/symbolic_transformations/utils.hpp"

using namespace ov::symbol::util;

ov::pass::DeReshapeConcat::DeReshapeConcat() {
    MATCHER_SCOPE(DeReshapeConcat);

    auto reshape_in0_m = pattern::wrap_type<ov::op::v1::Reshape>({pattern::any_input(pattern::has_static_rank()), pattern::wrap_type<ov::op::v0::Constant>()});
    auto reshape_in1_m = pattern::wrap_type<ov::op::v1::Reshape>({pattern::any_input(pattern::has_static_rank()), pattern::wrap_type<ov::op::v0::Constant>()});
    auto concat_m = pattern::wrap_type<ov::op::v0::Concat>({reshape_in0_m, reshape_in1_m});
    auto reshape_out_m = pattern::wrap_type<ov::op::v1::Reshape>({concat_m, pattern::wrap_type<ov::op::v0::Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pm = m.get_pattern_map();

        const auto& in0_reshape = pm.at(reshape_in0_m);
        const auto& in1_reshape = pm.at(reshape_in1_m);
        const auto& out_reshape = pm.at(reshape_out_m);
        const auto& concat = std::dynamic_pointer_cast<ov::op::v0::Concat>(pm.at(concat_m));

        const auto& in0_shape = in0_reshape->get_input_partial_shape(0);
        const auto& in0_target_shape = in0_reshape->get_output_partial_shape(0);
        const auto& in1_shape = in1_reshape->get_input_partial_shape(0);
        const auto& in1_target_shape = in1_reshape->get_output_partial_shape(0);
        const auto& out_shape = out_reshape->get_input_partial_shape(0);
        const auto& out_target_shape = out_reshape->get_output_partial_shape(0);

        const auto concat_axis = ov::util::normalize(concat->get_axis(), in0_target_shape.size());
        std::cerr << "try dereshape concat!\n";

        if (in0_shape.size() != out_target_shape.size() || in1_shape.size() != out_target_shape.size())
            return false;

        if (static_cast<int64_t>(in0_shape.size()) <= concat_axis || static_cast<int64_t>(in1_shape.size()) <= concat_axis)
            return false;

        if (!dims_are_equal(in0_target_shape[concat_axis], in0_shape[concat_axis]) ||
            !dims_are_equal(in1_target_shape[concat_axis], in1_shape[concat_axis]) ||
            !dims_are_equal(out_shape[concat_axis], out_target_shape[concat_axis]))
            return false;

        for (size_t i = 0; i < out_target_shape.size(); i++) {
            if (static_cast<int64_t>(i) == concat_axis)
                continue;

            if (!dims_are_equal(in0_shape[i], out_target_shape[i]) || !dims_are_equal(in1_shape[i], out_target_shape[i]))
                return false;
        }

        in0_reshape->output(0).replace(in0_reshape->input_value(0));
        in1_reshape->output(0).replace(in1_reshape->input_value(0));

        ov::replace_output_update_name(out_reshape->output(0), concat->output(0));
        concat->set_axis(concat_axis); // if the axis was negative, it could become invalid due to rank change. Set normalized one
        concat->validate_and_infer_types();

        std::cerr << "success!\n";

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reshape_out_m, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
