// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include "openvino/core/except.hpp"

namespace ov {
namespace intel_gpu {

struct Argument {
    static Argument input(size_t id) {
        OPENVINO_ASSERT(id < max_inputs_size);
        return Argument(inputs_offset + id);
    }

    static Argument output(size_t id) {
        OPENVINO_ASSERT(id < max_outputs_size);
        return Argument(outputs_offset + id);
    }

    static Argument weights() {
        return Argument(weights_offset);
    }

    static Argument bias() {
        return Argument(bias_offset);
    }

    static Argument post_op(size_t id) {
        return Argument(post_op_offset);
    }

    operator size_t() const { return m_arg_id; }

    Argument(const Argument& other) = default;
    Argument(Argument&& other) = default;
    Argument& operator=(const Argument& other) = default;
    Argument& operator=(Argument&& other) = default;

private:
    size_t m_arg_id;
    Argument(size_t id) : m_arg_id(id) {}

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

}  // namespace op
}  // namespace ov
