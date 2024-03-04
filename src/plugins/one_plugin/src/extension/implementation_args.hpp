// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <sstream>
#include "openvino/core/except.hpp"

namespace ov {

enum class ArgumentType : uint8_t {
    INPUT = 0,
    OUTPUT = 1,
    WEIGHTS = 2,
    BIAS = 3,
    POST_OP_INPUT = 4,
    UNDEF = 255,
};


inline std::ostream& operator<<(std::ostream& os, const ArgumentType& arg_type) {
    switch (arg_type) {
        case ArgumentType::INPUT: return os << "input";
        case ArgumentType::OUTPUT: return os << "output";
        case ArgumentType::WEIGHTS: return os << "weights";
        case ArgumentType::BIAS: return os << "bias";
        case ArgumentType::POST_OP_INPUT: return os << "post_op_input";
        case ArgumentType::UNDEF: return os << "undef";
    }
    return os;
}

struct Argument {
    static Argument input(size_t id) {
        OPENVINO_ASSERT(id < max_inputs_size);
        static const ArgumentType arg_type = ArgumentType::INPUT;
        return Argument(id, inputs_offset, arg_type);
    }

    static Argument output(size_t id) {
        OPENVINO_ASSERT(id < max_outputs_size);
        static const ArgumentType arg_type = ArgumentType::OUTPUT;
        return Argument(id, outputs_offset, arg_type);
    }

    static Argument weights() {
        static const ArgumentType arg_type = ArgumentType::WEIGHTS;
        return Argument(0, weights_offset, arg_type);
    }

    static Argument bias() {
        static const ArgumentType arg_type = ArgumentType::BIAS;
        return Argument(0, bias_offset, arg_type);
    }

    static Argument post_op(size_t id) {
        static const ArgumentType arg_type = ArgumentType::POST_OP_INPUT;
        return Argument(id, post_op_offset, arg_type);
    }

    operator size_t() const { return m_id + m_arg_offset; }

    ArgumentType type() const { return m_arg_type; }
    size_t id() const { return m_id; }

    Argument() = delete;
    Argument(const Argument& other) = default;
    Argument(Argument&& other) = default;
    Argument& operator=(const Argument& other) = default;
    ~Argument() = default;

    std::string to_string() const {
        std::stringstream s;
        s << m_arg_type << "(" << m_id << ")";
        return s.str();
    }

private:
    size_t m_id;
    size_t m_arg_offset;
    const ArgumentType m_arg_type;
    Argument(size_t id, size_t arg_offset, const ArgumentType& arg_type) : m_id(id), m_arg_offset(arg_offset), m_arg_type(arg_type) {}

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
