// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "select_implementations.hpp"

#include "gpu_opset/gpu_opset.hpp"
#include "implementation_map.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/format.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/op/reorder.hpp"
#include "intel_gpu/runtime/kernel.hpp"
#include "intel_gpu/runtime/lru_cache.hpp"
#include "openvino/core/rt_info.hpp"

#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "transformations/insert_reorders.hpp"
#include "transformations/utils/utils.hpp"
#include "gpu_opset/node_extension.hpp"
#include "layout_optimizer.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {

// namespace {
// cldnn::shape_types get_shape_type(const cldnn::kernel_impl_params& impl_params) {
//     for (auto& in_shape : impl_params.input_layouts) {
//         if (in_shape.is_dynamic()) {
//             return cldnn::shape_types::dynamic_shape;
//         }
//     }
//     if (impl_params.get_output_layout().is_dynamic())
//         return cldnn::shape_types::dynamic_shape;

//     return cldnn::shape_types::static_shape;
// }
// } // namespace

bool SelectImplementations::run_on_model(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ordered_ops()) {
        auto node = std::dynamic_pointer_cast<NodeExtension>(op);
        OPENVINO_ASSERT(node != nullptr);

        std::cerr << "SelectImplementations: handle " << op->get_friendly_name() << std::endl;

        node->select_best_implementation();

    }

    return false;
}

}  // namespace intel_gpu
}  // namespace ov
