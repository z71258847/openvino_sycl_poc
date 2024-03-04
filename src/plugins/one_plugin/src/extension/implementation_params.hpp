// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include "openvino/core/node.hpp"

namespace ov {

struct ImplementationParameters {
    explicit ImplementationParameters(const ov::Node* node) : m_node(node) {}
    std::string some_parameter = "";
    virtual ~ImplementationParameters() = default;

    const ov::Node* m_node;
};

}  // namespace ov
