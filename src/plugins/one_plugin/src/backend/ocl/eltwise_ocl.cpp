// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_ocl.hpp"
#include "extension/executor.hpp"
#include "impls/eltwise.hpp"

namespace ov {
namespace ocl {

class SomeEltwiseOCLExecutor : public OpExecutor {
public:
    explicit SomeEltwiseOCLExecutor(const EltwiseParams* params) : m_params(params) { }

    Event::Ptr execute(Stream& stream, const MemoryArgs& args, const Events dep_events) override {
        std::cerr << "SomeEltwiseOCLExecutor::execute()" << (int)m_params->type << "\n";
        return nullptr;
    }

private:
    const EltwiseParams* m_params;
};


bool SomeEltwiseOCLImpl::supports(const ImplementationParameters* params) const {
    return true;
}

OpExecutor::Ptr SomeEltwiseOCLImpl::get_executor() const {
    auto typed_params = dynamic_cast<const EltwiseParams*>(m_params);
    return std::make_shared<SomeEltwiseOCLExecutor>(typed_params);
}

void SomeEltwiseOCLImpl::init_kernel_data(const ImplementationParameters* params) {
    m_kernel_data = {};
}

}  // namespace ocl
}  // namespace ov
