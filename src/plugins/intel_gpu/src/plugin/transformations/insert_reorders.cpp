// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "insert_reorders.hpp"

#include "joint_impl/extended_opset.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/op/reorder.hpp"
#include "openvino/core/rt_info.hpp"

#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "transformations/insert_reorders.hpp"
#include "transformations/utils/utils.hpp"
#include "joint_impl/node_extension.hpp"
#include "layout_optimizer.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {

bool InsertReorders::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // for (const auto& op : model->get_ordered_ops()) {
    //     auto n = std::dynamic_pointer_cast<GPUOpExtension>(op);
    //     OPENVINO_ASSERT(n != nullptr);

    //     std::cerr << "handle " << op->get_friendly_name() << std::endl;
    //     for (size_t i = 0; i < op->get_input_size(); i++) {
    //         auto input = op->get_input_source_output(i);
    //         auto index = input.get_index();
    //         auto in_node = input.get_node_shared_ptr();
    //         auto input_node = std::dynamic_pointer_cast<GPUOpExtension>(in_node);
    //         auto src_fmt = input_node->get_preferred_output_fmt(index);
    //         auto dst_fmt = n->get_preferred_input_fmt(i);
    //         if (dst_fmt != src_fmt) {
    //             std::cerr << "FMT mismatch between "
    //                             << in_node->get_friendly_name() << "(" << index << "): " << src_fmt << " ---> "
    //                             << op->get_friendly_name() << "(" << i << ") " << dst_fmt << std::endl;
    //             auto reorder = make_gpu_op<op::Reorder>(in_node->output(index), dst_fmt);
    //             m_optimizer.select_preferred_formats(reorder);
    //             op->input(i).replace_source_output(reorder->output(0));
    //         }
    //     }
    // }

    return false;
}

}  // namespace intel_gpu
}  // namespace ov
