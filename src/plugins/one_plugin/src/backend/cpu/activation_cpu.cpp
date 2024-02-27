// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_cpu.hpp"
#include "extension/executor.hpp"
#include "impls/activation.hpp"

namespace ov {
namespace cpu {

class SomeActivationCPUExecutor : public OpExecutor {
public:
    explicit SomeActivationCPUExecutor(const ActivationParams* params) : m_params(params) {

    }

    void execute() override {
        std::cerr << "SomeActivationCPUExecutor::execute()" << (int)m_params->type << "\n";
    }

private:
    const ActivationParams* m_params;
};


bool SomeActivationCPUImpl::supports(const ImplementationParameters* params) const {
    return true;
}

OpExecutor::Ptr SomeActivationCPUImpl::get_executor(const ImplementationParameters* params) const {
    auto typed_params = dynamic_cast<const ActivationParams*>(params);
    return std::make_shared<SomeActivationCPUExecutor>(typed_params);
}

}  // namespace cpu
}  // namespace ov
