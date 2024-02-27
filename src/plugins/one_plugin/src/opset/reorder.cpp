// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

Reorder::Reorder(const ov::Output<Node>& input,
                 const Format output_format,
                 const ov::element::Type output_type)
    : Op({input}), m_output_format(output_format), m_output_type(output_type) {
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> Reorder::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<Reorder>(new_args.at(0), m_output_format, m_output_type);
}

void Reorder::validate_and_infer_types() {
    auto output_type = m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, get_input_partial_shape(0));
}

bool Reorder::visit_attributes(ov::AttributeVisitor &visitor) {
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("output_format", m_output_type);
    return true;
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
