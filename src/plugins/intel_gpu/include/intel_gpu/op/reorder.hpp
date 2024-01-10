// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

#include "intel_gpu/runtime/format.hpp"

namespace ov {
namespace intel_gpu {
using Format = cldnn::format;
namespace op {

class Reorder : public ov::op::Op {
public:
    OPENVINO_OP("Reorder", "gpu_opset");

    Reorder() = default;

    Reorder(const ov::Output<Node>& input,
            const Format output_format,
            const ov::element::Type output_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    Format get_output_format() const { return m_output_format; }
    ov::element::Type get_output_type() const { return m_output_type; }

protected:
    Format m_output_format = Format::any;
    ov::element::Type m_output_type;
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
