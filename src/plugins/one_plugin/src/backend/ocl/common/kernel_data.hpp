// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace ocl {

struct KernelString {
    std::string str;
    std::string jit;
    std::string undefs;
    std::string options;
    std::string entry_point;
    bool batch_compilation;

    KernelString() : str(""), jit(""), undefs(""), options(""), entry_point(""), batch_compilation(false) {}

    std::string get_str() const { return str + jit + undefs + options + entry_point; }
    size_t get_hash() const { return std::hash<std::string>()(get_str()); }
};

struct KernelCode {
    std::shared_ptr<KernelString> kernelString;
};

struct WorkGroups {
    std::vector<size_t> global;
    std::vector<size_t> local;
};

struct Argument {
    enum class Types {
        INPUT,
        OUTPUT,
        WEIGHTS,
        BIAS,
        SCALE_TABLE,
        SLOPE,
        INTERNAL_BUFFER,
        SCALAR,
        CELL,       // LSTM cell input
        WEIGHTS_ZERO_POINTS,
        ACTIVATIONS_ZERO_POINTS,
        COMPENSATION,
        INPUT_OF_FUSED_PRIMITIVE,
        SHAPE_INFO
    };

    Types t;
    uint32_t index;
};

using Arguments = std::vector<Argument>;

struct Scalar {
    union ValueT {
        uint8_t u8;
        uint16_t u16;
        uint32_t u32;
        uint64_t u64;
        int8_t s8;
        int16_t s16;
        int32_t s32;
        int64_t s64;
        float f32;
        double f64;
    };

    enum class Types {
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        INT8,
        INT16,
        INT32,
        INT64,
        FLOAT32,
        FLOAT64,
    };

    Types t;
    ValueT v;
};

using Scalars = std::vector<Scalar>;

struct KernelParams {
    WorkGroups workGroups;
    Arguments arguments;
    Scalars scalars;
    std::string layerID;
};

struct KernelData {
    KernelCode code;
    KernelParams params;
    bool skip_execution = false;
};

}  // namespace ocl
}  // namespace ov
