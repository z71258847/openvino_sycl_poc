// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "plugin.hpp"
#include "remote_context.hpp"
#include "execution_config.hpp"
#include "openvino/runtime/icompiled_model.hpp"

namespace ov {
namespace intel_gpu {

class CompiledModel : public ov::ICompiledModel {
public:
    using Ptr = std::shared_ptr<CompiledModel>;

    CompiledModel(std::shared_ptr<ov::Model> model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  RemoteContextImpl::Ptr context,
                  const ExecutionConfig& config);

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    ov::Any get_property(const std::string& name) const override;

    void set_property(const ov::AnyMap& properties) override {
        OPENVINO_THROW_NOT_IMPLEMENTED("It's not possible to set property of an already compiled model. Set property "
                                       "to Core::compile_model during compilation");
    };

    RemoteContextImpl::Ptr get_context_impl() const {
        return m_context;
    }

    std::shared_ptr<ov::Model> get_model() const { return m_model; }

private:
    RemoteContextImpl::Ptr m_context;
    ExecutionConfig m_config;
    std::shared_ptr<ov::threading::ITaskExecutor> m_wait_executor;
    std::string m_model_name;
    std::shared_ptr<ov::Model> m_model;
    bool m_loaded_from_cache;
};

}  // namespace intel_gpu
}  // namespace ov
