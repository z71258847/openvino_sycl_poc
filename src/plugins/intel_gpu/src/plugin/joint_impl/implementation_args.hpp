// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include "openvino/core/except.hpp"

namespace ov {

struct Argument {
    static Argument input(size_t id) {
        OPENVINO_ASSERT(id < max_inputs_size);
        static const std::string arg_type = "input";
        return Argument(id, inputs_offset, arg_type);
    }

    static Argument output(size_t id) {
        OPENVINO_ASSERT(id < max_outputs_size);
        static const std::string arg_type = "output";
        return Argument(id, outputs_offset, arg_type);
    }

    static Argument weights() {
        static const std::string arg_type = "weights";
        return Argument(0, weights_offset, arg_type);
    }

    static Argument bias() {
        static const std::string arg_type = "bias";
        return Argument(0, bias_offset, arg_type);
    }

    static Argument post_op(size_t id) {
        static const std::string arg_type = "post_op";
        return Argument(id, post_op_offset, arg_type);
    }

    operator size_t() const { return m_id + m_arg_offset; }

    Argument() = delete;
    Argument(const Argument& other) = default;
    Argument(Argument&& other) = default;
    Argument& operator=(const Argument& other) = default;
    Argument& operator=(Argument&& other) = default;
    ~Argument() = default;

    std::string to_string() const {
        return m_arg_type + "(" + std::to_string(m_id) + ")";
    }

private:
    size_t m_id;
    size_t m_arg_offset;
    const std::string m_arg_type;
    Argument(size_t id, size_t arg_offset, const std::string& arg_type) : m_id(id), m_arg_offset(arg_offset), m_arg_type(arg_type) {}

    static constexpr const size_t max_inputs_size = 32;
    static constexpr const size_t max_outputs_size = 32;
    static constexpr const size_t max_weights_size = 1;
    static constexpr const size_t max_bias_size = 1;

    static constexpr const size_t inputs_offset = 0;
    static constexpr const size_t outputs_offset = max_inputs_size;
    static constexpr const size_t weights_offset = max_inputs_size + max_outputs_size;
    static constexpr const size_t bias_offset = max_inputs_size + max_outputs_size + weights_offset;
    static constexpr const size_t post_op_offset = max_inputs_size + max_outputs_size + weights_offset + max_bias_size;
};

inline std::ostream& operator<<(std::ostream& os, const Argument& arg) {
    os << arg.to_string();
    return os;
}

}  // namespace ov
