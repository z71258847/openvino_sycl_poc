// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations_pipeline.hpp"

#include <memory>
#include <vector>

#include "common_transformations.hpp"
#include "convert_to_infer_precision.hpp"
#include "einsum_decomposition.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "lpt.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/pass/manager.hpp"
#include "optimize_with_internal_opset.hpp"
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/control_flow/unroll_tensor_iterator.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"
#include "transformations/resolve_names_collisions.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

namespace ov {
namespace intel_gpu {

bool TransformationsPipeline::run_on_model(const std::shared_ptr<ov::Model>& model) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TransformationsPipeline::run_on_model");
    using namespace ov::pass;
    Manager manager;
    auto pass_config = manager.get_pass_config();
    manager.set_per_pass_validation(false);

    // Temporary solution, global rt info cleanup is needed
    for (auto& node : model->get_ops()) {
        ov::enable_constant_folding(node);
        ov::disable_keep_const_precision(node);
    }

    if (m_context.run_lpt()) {
        const std::vector<ov::element::Type> supported_dequantization_precision = {ov::element::i8,
                                                                                   ov::element::u8,
                                                                                   ov::element::i4,
                                                                                   ov::element::u4};
        manager.register_pass<MarkDequantizationSubgraph>(supported_dequantization_precision);
    }

    manager.register_pass<InitNodeInfo>();
    manager.register_pass<EinsumDecomposition>();
    manager.register_pass<ConvertToInferPrecision>(m_context);
    manager.register_pass<CommonTransformations>(m_context);

    if (m_context.run_lpt()) {
        pass_config->disable<ConvertQuantizeDequantize>();
        manager.register_pass<ov::intel_gpu::LowPrecisionTransformations>(m_context);
    }

    // This ConstantFolding pass is added to fold reshapes added for constant inputs on NMS internal operation which prevents upper-bound
    // calculation
    // TODO: check why we have these reshapes
    manager.register_pass<ConstantFolding>();

    manager.register_pass<ov::pass::UnrollTensorIterator>();
    pass_config->set_callback<ov::pass::UnrollTensorIterator>([this](const std::shared_ptr<const ov::Node>& node) -> bool {
        auto sub_graph_op = std::dynamic_pointer_cast<const ov::op::util::SubGraphOp>(node);
        int64_t num_iter = sub_graph_op->get_num_iterations();
        if (!m_context.unroll_loop())
            return num_iter != 1;
        return num_iter >= 16;
    });
    manager.register_pass<OptimizeWithInternalOpset>(m_context);

    // This is supposed to be the last pass to ensure that we don't have name collisions until
    // GPU plugin stops using friendly names for program creation
    manager.register_pass<ResolveNameCollisions>(true);
    return manager.run_passes(model);
}
}  // namespace intel_gpu
}  // namespace ov
