// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_cpu.hpp"
#include "extension/executor.hpp"
#include "impls/eltwise.hpp"

namespace ov {
namespace cpu {

class SomeEltwiseCPUExecutor : public OpExecutor {
public:
    explicit SomeEltwiseCPUExecutor(const EltwiseParams* params) : m_params(params) {

    }

    Event::Ptr execute(Stream& stream, const MemoryArgs& args, const Events dep_events) override {
        std::cerr << "SomeEltwiseCPUExecutor::execute()" << (int)m_params->type << "\n";
        return nullptr;
    }

private:
    const EltwiseParams* m_params;
};


bool SomeEltwiseCPUImpl::supports(const ImplementationParameters* params) const {
    return true;
}

OpExecutor::Ptr SomeEltwiseCPUImpl::get_executor() const {
    auto typed_params = dynamic_cast<const EltwiseParams*>(m_params);
    return std::make_shared<SomeEltwiseCPUExecutor>(typed_params);
}

}  // namespace cpu
}  // namespace ov
