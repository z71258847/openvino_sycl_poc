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

REGISTER_OP_1(Parameter, op::v0::Parameter, FactoryParameters, NopImplementationsRegistry);
REGISTER_OP_1(Result, op::v0::Result, FactoryParameters, NopImplementationsRegistry);
REGISTER_OP_1(Constant, op::v0::Constant, FactoryParameters, NopImplementationsRegistry);
REGISTER_OP_1(Reshape, op::v1::Reshape, FactoryParameters, NopImplementationsRegistry);
REGISTER_OP_1(Placeholder, intel_gpu::op::Placeholder, FactoryParameters, NopImplementationsRegistry);

// REGISTER_OP_WITH_CUSTOM_PARAMS_AND_REGISTRY(Parameter_v0, ov::op::v0::Parameter, FactoryParameters, NopImplementationsRegistry);
// REGISTER_OP_WITH_CUSTOM_PARAMS_AND_REGISTRY(Result_v0, ov::op::v0::Result, FactoryParameters, NopImplementationsRegistry);
// REGISTER_OP_WITH_CUSTOM_PARAMS_AND_REGISTRY(Constant_v0, ov::op::v0::Constant, FactoryParameters, NopImplementationsRegistry);
// REGISTER_OP_WITH_CUSTOM_PARAMS_AND_REGISTRY(Placeholder_internal, ov::intel_gpu::op::Placeholder, FactoryParameters, NopImplementationsRegistry);
// REGISTER_OP_WITH_CUSTOM_PARAMS_AND_REGISTRY(Reshape_v1, ov::op::v1::Reshape, FactoryParameters, NopImplementationsRegistry);

}  // namespace ov
