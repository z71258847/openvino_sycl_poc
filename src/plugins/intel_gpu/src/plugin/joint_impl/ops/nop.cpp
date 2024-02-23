// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "joint_impl/executor.hpp"
#include "joint_impl/extended_opset.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "joint_impl/implementation_params.hpp"
#include "joint_impl/implementation_registry.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"

namespace ov {

class NopExecutor : public OpExecutor {
public:
    void execute() override { std::cerr << "NopExecutor::Execute()\n"; }
};

class NopImpl : public OpImplementation {
public:
    NopImpl() : OpImplementation("NopImpl") {}

    OpExecutor::Ptr get_executor(const ImplementationParameters*) const override { return std::make_shared<NopExecutor>(); }
    bool supports(const ImplementationParameters*) const override { return true; }
};

struct NopImplementationsRegistry : public ImplementationsRegistry<ImplementationParameters> {
    NopImplementationsRegistry() {
        register_impl<NopImpl>();
    }
    static const NopImplementationsRegistry& instance() {
        static NopImplementationsRegistry instance;
        return instance;
    }
};

REGISTER_OP(Parameter, op::v0::Parameter, ImplementationParameters, NopImplementationsRegistry);
REGISTER_OP(Result, op::v0::Result, ImplementationParameters, NopImplementationsRegistry);
REGISTER_OP(Constant, op::v0::Constant, ImplementationParameters, NopImplementationsRegistry);
REGISTER_OP(Reshape, op::v1::Reshape, ImplementationParameters, NopImplementationsRegistry);
REGISTER_OP(Placeholder, intel_gpu::op::Placeholder, ImplementationParameters, NopImplementationsRegistry);

}  // namespace ov
