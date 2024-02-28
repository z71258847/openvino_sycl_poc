// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extension/executor.hpp"
#include "extension/extended_opset.hpp"
#include "opset/placeholder.hpp"
#include "extension/implementation_params.hpp"
#include "extension/implementation_registry.hpp"
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

    OpExecutor::Ptr get_executor() const override { return std::make_shared<NopExecutor>(); }
    bool supports(const ImplementationParameters*) const override { return true; }
    void initialize(const ImplementationParameters*) override { }
};

struct NopImplementationsRegistry : public ImplementationsRegistry {
    NopImplementationsRegistry() {
        register_impl<NopImpl>();
    }
    static const NopImplementationsRegistry& instance() {
        static NopImplementationsRegistry instance;
        return instance;
    }
};

REGISTER_IMPLS(Parameter, op::v0::Parameter, ImplementationParameters, NopImplementationsRegistry);
REGISTER_IMPLS(Result, op::v0::Result, ImplementationParameters, NopImplementationsRegistry);
REGISTER_IMPLS(Constant, op::v0::Constant, ImplementationParameters, NopImplementationsRegistry);
REGISTER_IMPLS(Reshape, op::v1::Reshape, ImplementationParameters, NopImplementationsRegistry);
REGISTER_IMPLS(Placeholder, intel_gpu::op::Placeholder, ImplementationParameters, NopImplementationsRegistry);

}  // namespace ov
