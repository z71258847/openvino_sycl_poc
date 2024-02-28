// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_to_space_cpu.hpp"
#include "extension/executor.hpp"
#include "impls/batch_to_space.hpp"

namespace ov {
namespace cpu {

class SomeBatchToSpaceCPUExecutor : public OpExecutor {
public:
    explicit SomeBatchToSpaceCPUExecutor(const BatchToSpaceParams* params) : m_params(params) { }

    void execute() override {
        std::cerr << "SomeBatchToSpaceCPUExecutor::execute() " << m_params->some_parameter << "\n";
    }

private:
    const BatchToSpaceParams* m_params;
};

bool SomeBatchToSpaceCPUImpl::supports(const ImplementationParameters* params) const {
    return true;
}

OpExecutor::Ptr SomeBatchToSpaceCPUImpl::get_executor() const {
    auto typed_params = dynamic_cast<const BatchToSpaceParams*>(m_params);
    return std::make_shared<SomeBatchToSpaceCPUExecutor>(typed_params);
}

}  // namespace cpu
}  // namespace ov
