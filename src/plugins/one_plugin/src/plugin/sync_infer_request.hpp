// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"
#include "variable_state.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "remote_tensor.hpp"

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <atomic>

namespace ov {
namespace intel_gpu {

class CompiledModel;

class SyncInferRequest : public ov::ISyncInferRequest {
public:
    using Ptr = std::shared_ptr<SyncInferRequest>;

    explicit SyncInferRequest(const std::shared_ptr<const CompiledModel>& compiled_model);
    SyncInferRequest(const SyncInferRequest &) = delete;
    ~SyncInferRequest() override = default;

    void infer() override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    // void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    // void set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    // ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

private:
    void check_tensors() const override;

    VariablesMap m_variables;

    std::shared_ptr<ov::threading::IStreamsExecutor> m_stream_executor = nullptr;
};

}  // namespace intel_gpu
}  // namespace ov
