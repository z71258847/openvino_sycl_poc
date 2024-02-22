// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "select_implementations.hpp"

#include "joint_impl/node_extension.hpp"

#include <memory>

namespace ov {

bool SelectImplementations::run_on_model(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ordered_ops()) {
        auto node = std::dynamic_pointer_cast<NodeExtension>(op);
        OPENVINO_ASSERT(node != nullptr);

        std::cerr << "SelectImplementations: handle " << op->get_friendly_name() << std::endl;

        node->select_best_implementation();

        OPENVINO_ASSERT(node->get_impl() != nullptr, "No impl selected for: ", op->get_friendly_name());
    }

    return false;
}

}  // namespace ov
