// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include "openvino/core/node.hpp"

namespace ov {

struct FactoryParameters {
    explicit FactoryParameters(const ov::Node* node = nullptr) {}
    std::string some_parameter = "";
    virtual ~FactoryParameters() = default;
};

template <typename NodeType>
struct TypedNodeParams : public FactoryParameters {
    explicit TypedNodeParams(const NodeType* node) {}
    TypedNodeParams() = default;
};

}  // namespace ov
