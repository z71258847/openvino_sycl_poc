// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layout_optimizer.hpp"
#include "extension/memory_descriptor.hpp"
#include "extension/op_implementation.hpp"
#include "node_extension.hpp"

namespace ov {

std::vector<Configuration> LayoutOptimizer::get_available_configurations(const NodeExtension* op) const {
    return {
        Configuration{op->get_default_descriptors(), OpImplementation::Type::CPU, ConfigurationProperties{}}
    };
}

}  // namespace ov
