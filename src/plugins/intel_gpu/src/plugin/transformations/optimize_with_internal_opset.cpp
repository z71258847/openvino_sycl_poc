// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "optimize_with_internal_opset.hpp"

#include <memory>

#include "binary_conv_to_conv.hpp"
#include "clamp_fp16_output.hpp"
#include "convert_fc_to_compressed.hpp"
#include "convert_gather_to_compressed.hpp"
#include "convert_matmul_to_fc.hpp"
#include "fc_convert_fusion.hpp"
#include "indirect_kv_cache.hpp"
#include "kv_cache_fusion.hpp"
#include "move_fc_reshape_to_weights.hpp"
#include "openvino/pass/manager.hpp"
#include "rms_fusion.hpp"
#include "swiglu_fusion.hpp"
#include "transpose_matmul_fusion.hpp"

namespace ov {
namespace intel_gpu {

bool OptimizeWithInternalOpset::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::intel_gpu::ConvertBinaryConvolutionToConvolution>();
    manager.register_pass<ov::intel_gpu::ClampFP16Output>();
    manager.register_pass<ov::intel_gpu::ConvertMatMulToFullyConnected>();
    manager.register_pass<ov::intel_gpu::MoveFCReshapeToWeights>();
    manager.register_pass<ov::intel_gpu::ConvertFullyConnectedToFullyConnectedCompressed>();
    manager.register_pass<ov::intel_gpu::ConvertGatherToGatherCompressed>();
    manager.register_pass<ov::intel_gpu::RMSFusion>(m_context.get_device_info().max_work_group_size);
    manager.register_pass<ov::intel_gpu::KVCacheFusion>();
    manager.register_pass<ov::intel_gpu::FullyConnectedConvertFusion>();
    if (!m_context.has_dpas())
        manager.register_pass<ov::intel_gpu::TransposeMatMulFusion>();
    manager.register_pass<ov::intel_gpu::SwiGLUFusion>();

    manager.register_pass<ov::intel_gpu::IndirectKVCache>();
    return manager.run_passes(model);
}
}  // namespace intel_gpu
}  // namespace ov
