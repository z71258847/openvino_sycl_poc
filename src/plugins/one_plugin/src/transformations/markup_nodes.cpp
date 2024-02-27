// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "markup_nodes.hpp"

#include "extension/extended_opset.hpp"
#include "openvino/core/rt_info.hpp"

#include <memory>
#include <deque>
#include <unordered_set>

namespace ov {
namespace intel_gpu {

bool MarkupNodes::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // std::deque<ov::Node*> nodes;
    // std::unordered_set<ov::Node*> visited;

    // for (const auto& r : model->get_results()) {
    //     nodes.push_back(r.get());
    //     visited.insert(r.get());
    // }
    // for (const auto& r : model->get_sinks()) {
    //     nodes.emplace_back(r.get());
    //     visited.insert(r.get());
    // }

    // while (!nodes.empty()) {
    //     auto curr_node = nodes.front();
    //     nodes.pop_front();
    //     for (const auto& input : curr_node->inputs()) {
    //         if (ov::is_precision_sensitive(input)) {
    //             visited.insert(input.get_source_output().get_node());
    //             // visit_shape_path shouldn't depend on "visited" nodes because we can approach Divide
    //             // earlier from some non precision sensitive path. So we use dedicated "precision_sensitive_visited"
    //             // set for precision sensitive nodes, so they can be visited twice and finally marked-up.
    //             ov::op::util::visit_shape_path(input.get_source_output().get_node(),
    //                                            precision_sensitive_visited,
    //                                            m_markup_func);
    //         }
    //     }

    //     for (auto& input_value : curr_node->input_values()) {
    //         // continue searching
    //         const auto& input_node = input_value.get_node();
    //         if (visited.count(input_node))
    //             continue;

    //         if (auto sub_graph_node = ov::as_type<ov::op::util::MultiSubGraphOp>(input_node)) {
    //             size_t sub_graphs_num = sub_graph_node->get_internal_subgraphs_size();
    //             for (size_t sub_graph_ind = 0; sub_graph_ind < sub_graphs_num; ++sub_graph_ind) {
    //                 auto sub_graph = sub_graph_node->get_function(static_cast<int>(sub_graph_ind));
    //                 run_on_model(sub_graph);
    //             }
    //         }
    //         nodes.push_front(input_node);
    //         visited.insert(input_node);
    //     }
    // }
    return true;
}

}  // namespace intel_gpu
}  // namespace ov
