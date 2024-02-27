// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "asymmetric_convolution.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

AsymmetricConvolution::AsymmetricConvolution(const ov::Output<Node>& data_batch,
                                             const ov::Output<Node>& filters,
                                             const ov::Strides& strides,
                                             const ov::CoordinateDiff& pads_begin,
                                             const ov::CoordinateDiff& pads_end,
                                             const ov::Strides& dilations,
                                             const ov::op::PadType& auto_pad)
    : Convolution(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad) {
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> AsymmetricConvolution::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<AsymmetricConvolution>(new_args.at(0), new_args.at(1), m_strides, m_pads_begin, m_pads_end, m_dilations, m_auto_pad);
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
