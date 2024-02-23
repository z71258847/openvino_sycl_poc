// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_gpu.hpp"
#include "joint_impl/executor.hpp"
#include "joint_impl/ops/activation.hpp"

namespace ov {
namespace gpu {

class SomeActivationGPUExecutor : public OpExecutor {
public:
    explicit SomeActivationGPUExecutor(const ActivationParams* params) : m_params(params) {

    }

    void execute() override {
        std::cerr << "SomeActivationGPUExecutor::execute()" << (int)m_params->type << "\n";
    }

private:
    const ActivationParams* m_params;
};


bool SomeActivationGPUImpl::supports(const ImplementationParameters* params) const {
    return true;
}

OpExecutor::Ptr SomeActivationGPUImpl::get_executor(const ImplementationParameters* params) const {
    auto typed_params = dynamic_cast<const ActivationParams*>(params);
    return std::make_shared<SomeActivationGPUExecutor>(typed_params);
}

}  // namespace gpu
}  // namespace ov
