// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_cpu.hpp"
#include "extension/executor.hpp"
#include "impls/fully_connected.hpp"
#include "openvino/op/matmul.hpp"

namespace ov {
namespace cpu {

class SomeFullyConnectedCPUExecutor : public OpExecutor {
public:
    explicit SomeFullyConnectedCPUExecutor(const FullyConnectedParams* params) : m_params(params) { }

    Event::Ptr execute(Stream& stream, const MemoryArgs& args, const Events dep_events) override {
        std::cerr << "SomeFullyConnectedCPUExecutor::execute() " << (m_params != nullptr ? "with params" : "null params") << "\n";

        ov::TensorVector inputs {args.at(Argument::input(0))->to_tensor(), args.at(Argument::weights())->to_tensor()};
        ov::TensorVector outputs {args.at(Argument::output(0))->to_tensor()};
        OPENVINO_ASSERT(m_params != nullptr);
        OPENVINO_ASSERT(m_params->m_node != nullptr);
        ov::op::v0::MatMul op;
        op.set_transpose_a(false);
        op.set_transpose_b(true);
        OPENVINO_ASSERT(op.evaluate(outputs, inputs));

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
