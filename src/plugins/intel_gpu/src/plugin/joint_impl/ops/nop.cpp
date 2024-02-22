// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "joint_impl/extended_opset.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "joint_impl/implementation_params.hpp"
#include "joint_impl/implementation_registry.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"

namespace ov {

class NopImpl : public OpImplementation {
public:
    NopImpl(const FactoryParameters&) : OpImplementation("NopImpl") {}

    void execute() override {
        std::cerr << "NopImpl::execute()!\n";
    }
};

struct NopImplementationsRegistry : public ImplementationsRegistry<FactoryParameters> {
    NopImplementationsRegistry() {
        register_impl<NopImpl>();
    }
    static const NopImplementationsRegistry& instance() {
        static NopImplementationsRegistry instance;
        return instance;
    }
};

REGISTER_OP(Parameter, op::v0::Parameter, FactoryParameters, NopImplementationsRegistry);
REGISTER_OP(Result, op::v0::Result, FactoryParameters, NopImplementationsRegistry);
REGISTER_OP(Constant, op::v0::Constant, FactoryParameters, NopImplementationsRegistry);
REGISTER_OP(Reshape, op::v1::Reshape, FactoryParameters, NopImplementationsRegistry);
REGISTER_OP(Placeholder, intel_gpu::op::Placeholder, FactoryParameters, NopImplementationsRegistry);

}  // namespace ov
