// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace ov {
namespace intel_gpu {

struct FactoryParameters {
    std::string some_parameter = "";
};

template <typename NodeType>
struct TypedNodeParams : public FactoryParameters {
    explicit TypedNodeParams(const NodeType* node) {}
    TypedNodeParams() = default;
};

}  // namespace op
}  // namespace ov
