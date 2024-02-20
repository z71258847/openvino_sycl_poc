// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"

namespace ov {
namespace intel_gpu {

struct FactoryParameters { };

template <typename NodeType>
struct TypedNodeParams : FactoryParameters {
    std::string some_parameter = "";
};

template<>
struct TypedNodeParams<ov::op::v0::Parameter> : public FactoryParameters {
    int some_parameter = 0;
};

template<>
struct TypedNodeParams<ov::op::v0::MatMul> : public FactoryParameters {
    bool some_parameter = false;
};

}  // namespace op
}  // namespace ov
