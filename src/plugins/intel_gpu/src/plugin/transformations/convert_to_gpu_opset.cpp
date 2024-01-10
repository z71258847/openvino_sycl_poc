// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_to_gpu_opset.hpp"
#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/opsets/opset12.hpp"
#include "ov_ops/nms_ie_internal.hpp"
#include "ov_ops/nms_static_shape_ie.hpp"
#include "ov_ops/multiclass_nms_ie_internal.hpp"
#include "ov_ops/generate_proposals_ie_internal.hpp"
#include "ov_ops/type_relaxed.hpp"

#include "gpu_opset/gpu_opset.hpp"
#include "gpu_opset/gpu_op_extension.hpp"

#include "intel_gpu/runtime/debug_configuration.hpp"

namespace {
void replace_node_unsafe(const std::shared_ptr<ov::Node>& target, const std::shared_ptr<ov::Node>& replacement) {
    OPENVINO_ASSERT(target->get_output_size() == replacement->get_output_size(),
                    "Target output size: ",
                    target->get_output_size(),
                    " must be equal replacement output size: ",
                    replacement->get_output_size());

    std::unordered_set<std::shared_ptr<ov::Node>> replacement_nodes;
    // For each of target's output O with replacement output O_rep:
    //     For each O's connected downstream input I:
    //         Change I's connected upstream output to O_rep
    for (size_t i = 0; i < target->get_output_size(); i++) {
        auto replacement_value = replacement->output(i);
        auto replacement_node = replacement_value.get_node_shared_ptr();
        if (replacement_nodes.find(replacement_node) == replacement_nodes.end()) {
            replacement_node->add_node_control_dependents(target);
            replacement_node->add_node_control_dependencies(target);
            replacement_nodes.insert(replacement_node);
        }
        target->output(i).replace(replacement_value);
    }
    target->clear_control_dependents();
    target->clear_control_dependents();
}
}  // namespace

namespace ov {
namespace intel_gpu {

ov::intel_gpu::ConvertToGpuOpset::ConvertToGpuOpset() {
}

bool ConvertToGpuOpset::run_on_model(const std::shared_ptr<ov::Model>& m) {
    for (auto& op : m->get_ordered_ops()) {
        auto gpu_op = gpu_op_converter().convert_to_gpu_opset(op);
        gpu_op->set_output_size(op->get_output_size());
        gpu_op->set_friendly_name(op->get_friendly_name());
        ov::copy_runtime_info(op, gpu_op);
        replace_node_unsafe(op, gpu_op);
        if (auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(op)) {
            m->remove_parameter(param);
            m->add_parameters({std::dynamic_pointer_cast<ov::op::v0::Parameter>(gpu_op)});
        } else if (auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(op)) {
            m->remove_result(result);
            m->add_results({std::dynamic_pointer_cast<ov::op::v0::Result>(gpu_op)});
        } else if (auto sink = std::dynamic_pointer_cast<ov::op::Sink>(op)) {
            m->remove_sink(sink);
            m->add_sinks({std::dynamic_pointer_cast<ov::op::Sink>(gpu_op)});
        }
    }

    return true;
}

}  // namespace intel_gpu
}  // namespace ov
