// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_conv_to_asymmetric_conv.hpp"
#include <memory>

#include "opset/asymmetric_convolution.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

ConvertConvolutionToAsymmetricConvolution::ConvertConvolutionToAsymmetricConvolution() {
    using namespace ov::pass::pattern;

    auto convolution_m = wrap_type<ov::op::v1::Convolution>({any_input(), any_input()});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto conv = std::dynamic_pointer_cast<ov::op::v1::Convolution>(pattern_map.at(convolution_m).get_node_shared_ptr());
        if (!conv || transformation_callback(conv)) {
            return false;
        }

        auto new_conv = std::make_shared<ov::intel_gpu::op::AsymmetricConvolution>();

        new_conv->set_friendly_name(conv->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), new_conv);
        ov::replace_node(conv, new_conv);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(convolution_m, "ConvertConvolutionToAsymmetricConvolution");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
