// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/convolution.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

class AsymmetricConvolution : public ov::op::v1::Convolution {
public:
    OPENVINO_OP("AsymmetricConvolution", "gpu_opset");

    AsymmetricConvolution() = default;

    AsymmetricConvolution(const ov::Output<Node>& data_batch,
                          const ov::Output<Node>& filters,
                          const ov::Strides& strides,
                          const ov::CoordinateDiff& pads_begin,
                          const ov::CoordinateDiff& pads_end,
                          const ov::Strides& dilations,
                          const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT);

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
