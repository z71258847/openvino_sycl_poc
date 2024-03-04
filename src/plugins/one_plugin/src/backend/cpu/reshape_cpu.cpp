// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_cpu.hpp"
#include "extension/executor.hpp"
#include "impls/reshape.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace cpu {

class SomeReshapeCPUExecutor : public OpExecutor {
public:
    explicit SomeReshapeCPUExecutor(const ReshapeParams* params) : m_params(params) {

    }

    Event::Ptr execute(Stream& stream, const MemoryArgs& args, const Events dep_events) override {
        std::cerr << "SomeReshapeCPUExecutor::execute()\n";

        auto input = args.at(Argument::input(0));
        auto output = args.at(Argument::output(0));
        std::memcpy(output->ptr, input->ptr, output->size());
        return nullptr;
    }

private:
    const ReshapeParams* m_params;
};


bool SomeReshapeCPUImpl::supports(const ImplementationParameters* params) const {
    return true;
}

OpExecutor::Ptr SomeReshapeCPUImpl::get_executor() const {
    auto typed_params = dynamic_cast<const ReshapeParams*>(m_params);
    return std::make_shared<SomeReshapeCPUExecutor>(typed_params);
}

}  // namespace cpu
}  // namespace ov
