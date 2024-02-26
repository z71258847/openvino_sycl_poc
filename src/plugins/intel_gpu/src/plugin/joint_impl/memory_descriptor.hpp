// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "joint_impl/implementation_args.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "intel_gpu/runtime/format.hpp"

namespace ov {

using Format = cldnn::format;

struct MemoryDesc {
    MemoryDesc()
        : m_format(Format::any)
        , m_data_type(ov::element::undefined)
        , m_shape(ov::PartialShape::dynamic())
        , m_pad_b(ov::PartialShape::dynamic())
        , m_pad_e(ov::PartialShape::dynamic()) {}

    explicit MemoryDesc(const Format& fmt)
        : m_format(fmt)
        , m_data_type(ov::element::undefined)
        , m_shape(ov::PartialShape::dynamic())
        , m_pad_b(ov::PartialShape::dynamic())
        , m_pad_e(ov::PartialShape::dynamic()) {}

    MemoryDesc(const Format& fmt, ov::PartialShape shape)
        : m_format(fmt)
        , m_data_type(ov::element::undefined)
        , m_shape(shape)
        , m_pad_b(ov::PartialShape::dynamic())
        , m_pad_e(ov::PartialShape::dynamic()) {}

    Format m_format;
    element::Type m_data_type;
    ov::PartialShape m_shape;
    ov::PartialShape m_pad_b; // need partialshape here ?
    ov::PartialShape m_pad_e; // need partialshape here ?
};

class MemoryDescs : public std::map<Argument, MemoryDesc> {
public:
    std::string to_string() const {
        std::string res;
        for (auto& kv : *this) {
            res += kv.first.to_string() + " " + kv.second.m_format.to_string() + ":" + kv.second.m_shape.to_string() + "\n";
        }

        return res;
    }

    bool has(const Argument& arg) const {
        return find(arg) != end();
    }
};

inline std::ostream& operator<<(std::ostream& os, const MemoryDescs& val) {
    os << val.to_string();
    return os;
}

}  // namespace ov
