// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_gpu.hpp"
#include "extension/executor.hpp"
#include "impls/eltwise.hpp"

namespace ov {
namespace gpu {

class SomeEltwiseGPUExecutor : public OpExecutor {
public:
    explicit SomeEltwiseGPUExecutor(const EltwiseParams* params) : m_params(params) { }

    void execute() override {
        std::cerr << "SomeEltwiseGPUExecutor::execute()" << (int)m_params->type << "\n";
    }

private:
    const EltwiseParams* m_params;
};


bool SomeEltwiseGPUImpl::supports(const ImplementationParameters* params) const {
    return true;
}

OpExecutor::Ptr SomeEltwiseGPUImpl::get_executor(const ImplementationParameters* params) const {
    auto typed_params = dynamic_cast<const EltwiseParams*>(params);
    return std::make_shared<SomeEltwiseGPUExecutor>(typed_params);
}

}  // namespace gpu
}  // namespace ov
