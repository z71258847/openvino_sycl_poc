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

enum class TensorOwner : uint8_t {
    USER = 0,
    PLUGIN = 1
};

struct TensorWrapper {
    TensorWrapper(const std::shared_ptr<ov::ITensor>& _ptr, TensorOwner _owner)
        : ptr(_ptr)
        , owner(_owner)
        , actual_size(_ptr ? _ptr->get_byte_size() : 0) {}

    TensorWrapper(const TensorWrapper& other) = default;
    TensorWrapper() = default;

    std::shared_ptr<ov::ITensor> ptr;
    TensorOwner owner;
    size_t actual_size;
};

class SyncInferRequest : public ov::ISyncInferRequest {
public:
    using Ptr = std::shared_ptr<SyncInferRequest>;

    explicit SyncInferRequest(const std::shared_ptr<const CompiledModel>& compiled_model);
    SyncInferRequest(const SyncInferRequest &) = delete;
    ~SyncInferRequest() override = default;

    void infer() override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    void set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    bool use_external_queue() const { return m_use_external_queue; }

private:
    void check_tensors() const override;

    std::unordered_map<std::string, TensorWrapper> m_user_inputs;
    std::unordered_map<std::string, TensorWrapper> m_user_outputs;

    std::unordered_map<std::string, TensorWrapper> m_plugin_inputs;
    std::unordered_map<std::string, TensorWrapper> m_plugin_outputs;

    std::unordered_map<std::string, ov::Output<const ov::Node>> m_input_ports_map;
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_output_ports_map;
    std::unordered_map<std::string, std::string> m_output_names_map;

    VariablesMap m_variables;

    std::shared_ptr<ov::threading::IStreamsExecutor> m_stream_executor = nullptr;
    bool m_enable_profiling = false;
    bool m_use_external_queue = false;
};

}  // namespace intel_gpu
}  // namespace ov
