// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_cpu.hpp"
#include "extension/executor.hpp"
#include "impls/convolution.hpp"

namespace ov {
namespace cpu {

class SomeConvolutionCPUExecutor : public OpExecutor {
public:
    explicit SomeConvolutionCPUExecutor(const SomeCustomParams* params) : m_params(params) { }

    void execute() override {
        std::cerr << "SomeConvolutionCPUExecutor::execute() " << (m_params != nullptr ? "with params" : "null params") << "\n";
    }

private:
    const SomeCustomParams* m_params;
};


bool SomeConvolutionCPUImpl::supports(const ImplementationParameters* params) const {
    return true;
}

OpExecutor::Ptr SomeConvolutionCPUImpl::get_executor() const {
    auto typed_params = dynamic_cast<const SomeCustomParams*>(m_params);
    return std::make_shared<SomeConvolutionCPUExecutor>(typed_params);
}

}  // namespace cpu
}  // namespace ov
