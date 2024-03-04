// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_cpu.hpp"
#include "extension/executor.hpp"
#include "impls/activation.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace cpu {

class SomeActivationCPUExecutor : public OpExecutor {
public:
    explicit SomeActivationCPUExecutor(const ActivationParams* params) : m_params(params) {

    }

    Event::Ptr execute(Stream& stream, const MemoryArgs& args, const Events dep_events) override {
        std::cerr << "SomeActivationCPUExecutor::execute()" << (int)m_params->type << "\n";

        auto input = args.at(Argument::input(0));
        auto output = args.at(Argument::output(0));
        ov::TensorVector inputs {input->to_tensor()};
        ov::TensorVector outputs {output->to_tensor()};
        OPENVINO_ASSERT(m_params != nullptr);
        OPENVINO_ASSERT(m_params->m_node != nullptr);
        OPENVINO_ASSERT(m_params->m_node->evaluate(outputs, inputs));

        return nullptr;
    }

private:
    const ActivationParams* m_params;
};


bool SomeActivationCPUImpl::supports(const ImplementationParameters* params) const {
    return true;
}

OpExecutor::Ptr SomeActivationCPUImpl::get_executor() const {
    auto typed_params = dynamic_cast<const ActivationParams*>(m_params);
    return std::make_shared<SomeActivationCPUExecutor>(typed_params);
}

}  // namespace cpu
}  // namespace ov
