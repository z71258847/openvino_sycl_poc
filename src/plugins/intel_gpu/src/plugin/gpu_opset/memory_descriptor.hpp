// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gpu_opset/implementation_args.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "intel_gpu/runtime/format.hpp"

namespace ov {
namespace intel_gpu {

using Format = cldnn::format;

struct MemoryDesc {
    MemoryDesc()
        : m_format(Format::any)
        , m_data_type(ov::element::undefined)
        , m_shape(ov::PartialShape::dynamic())
        , m_pad_b(ov::PartialShape::dynamic())
        , m_pad_e(ov::PartialShape::dynamic()) {}

    Format m_format;
    element::Type m_data_type;
    ov::PartialShape m_shape;
    ov::PartialShape m_pad_b; // need partialshape here ?
    ov::PartialShape m_pad_e; // need partialshape here ?
};

using MemoryDescs = std::map<Argument, MemoryDesc>;

}  // namespace op
}  // namespace ov
