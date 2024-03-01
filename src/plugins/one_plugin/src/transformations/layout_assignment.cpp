// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layout_assignment.hpp"
#include "extension/node_extension.hpp"

#include <memory>

namespace ov {

bool LayoutAssignment::run_on_model(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ordered_ops()) {
        auto node = std::dynamic_pointer_cast<NodeExtension>(op);
        node->select_preferred_formats(m_optimizer);

        std::cerr << "LayoutAssignment: " << op->get_friendly_name() << " " << op->get_type_name() << std::endl;
        for (auto& c : node->get_available_configurations()) {
            std::cerr << "config: " <<  c.m_desc << std::endl;

        }
    }

    return false;
}

}  // namespace ov
