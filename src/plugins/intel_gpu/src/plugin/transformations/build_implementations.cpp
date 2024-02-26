// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "build_implementations.hpp"

#include "joint_impl/node_extension.hpp"
#include "joint_impl/op_implementation.hpp"

#include <memory>

namespace ov {


bool BuildImplementations::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ImplementationBuilders cache;
    for (const auto& op : model->get_ordered_ops()) {
        auto node = std::dynamic_pointer_cast<NodeExtension>(op);

        cache.add_impl(node->get_impl());
    }

    cache.build();

    for (const auto& op : model->get_ordered_ops()) {
        auto node = std::dynamic_pointer_cast<NodeExtension>(op);

        node->create_executor(cache);
    }

    return false;
}

}  // namespace ov
