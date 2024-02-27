// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "joint_impl/memory_descriptor.hpp"

namespace ov {
class NodeExtension;

class LayoutOptimizer {
public:
    virtual void set_preferred_descriptors(const NodeExtension* op, MemoryDescs& current) const { }
};

}  // namespace ov
