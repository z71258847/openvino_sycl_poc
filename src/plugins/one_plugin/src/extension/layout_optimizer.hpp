// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "extension/memory_descriptor.hpp"

namespace ov {
class NodeExtension;

class LayoutOptimizer {
public:
    virtual std::vector<Configuration> get_available_configurations(const NodeExtension* op) const;
};

}  // namespace ov
