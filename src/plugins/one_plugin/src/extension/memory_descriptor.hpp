// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "extension/implementation_args.hpp"
#include "extension/op_implementation.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {

struct Format {
    enum Type {
        any = 0,
        bfyx = 1,
    };

    Type type;

    /// @brief Implicit conversion from format::type.
    constexpr Format(Type t) : type(t) {}
    /// @brief Implicit conversion to format::type.
    constexpr operator Type() const { return type; }

    std::string to_string() const { return type == any ? "any" : "not any"; }
};

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
    ov::PartialShape m_shape; // may be a custom class from CPU plugin for shape representation
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

struct ConfigurationProperties {
    bool shape_agnostic;

    std::vector<uint8_t> pad_begin_mask; // dynamic pad mask
    std::vector<uint8_t> pad_end_mask;
};

struct Configuration {
    MemoryDescs m_desc;
    OpImplementation::Type m_type;
    ConfigurationProperties m_properties;
};

inline std::ostream& operator<<(std::ostream& os, const MemoryDescs& val) {
    os << val.to_string();
    return os;
}

}  // namespace ov
