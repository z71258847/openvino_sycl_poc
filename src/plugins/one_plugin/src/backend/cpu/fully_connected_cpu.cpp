// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_cpu.hpp"
#include "extension/executor.hpp"
#include "impls/fully_connected.hpp"

namespace ov {
namespace cpu {

class SomeFullyConnectedCPUExecutor : public OpExecutor {
public:
    explicit SomeFullyConnectedCPUExecutor(const FullyConnectedParams* params) : m_params(params) { }

    Event::Ptr execute(Stream& stream, const MemoryArgs& args, const Events dep_events) override {
        std::cerr << "SomeFullyConnectedCPUExecutor::execute() " << (m_params != nullptr ? "with params" : "null params") << "\n";
        return nullptr;
    }

private:
    const FullyConnectedParams* m_params;
};


bool SomeFullyConnectedCPUImpl::supports(const ImplementationParameters* params) const {
    return true;
}

OpExecutor::Ptr SomeFullyConnectedCPUImpl::get_executor() const {
    auto typed_params = dynamic_cast<const FullyConnectedParams*>(m_params);
    return std::make_shared<SomeFullyConnectedCPUExecutor>(typed_params);
}

}  // namespace cpu
}  // namespace ov
