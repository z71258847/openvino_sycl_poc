// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/op_types.hpp"
#include "transformations/utils/utils.hpp"

#include "sync_infer_request.hpp"
#include "remote_context.hpp"
#include "remote_tensor.hpp"
#include "compiled_model.hpp"
#include "variable_state.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <utility>

namespace ov {
namespace intel_gpu {

// ----------------------------------------------------------------------------------------------- //
// ---------------------------- OpenVINO API impl ------------------------------------------------ //
// ----------------------------------------------------------------------------------------------- //

SyncInferRequest::SyncInferRequest(const std::shared_ptr<const CompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
}

void SyncInferRequest::infer() {
}

std::vector<ov::ProfilingInfo> SyncInferRequest::get_profiling_info() const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::vector<ov::SoPtr<ov::IVariableState>> SyncInferRequest::query_state() const {
    std::vector<ov::SoPtr<ov::IVariableState>> ret{};
    for (const auto& pair : m_variables) {
        ret.emplace_back(pair.second, nullptr);
    }
    return ret;
}

void SyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
}

void SyncInferRequest::set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    OPENVINO_THROW("[GPU] Cannot find input tensors for port ", port);
}

ov::SoPtr<ov::ITensor> SyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    OPENVINO_NOT_IMPLEMENTED;
}

void SyncInferRequest::check_tensors() const {
}

}  // namespace intel_gpu
}  // namespace ov
