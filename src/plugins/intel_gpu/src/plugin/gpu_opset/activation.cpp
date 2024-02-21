// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu_opset.hpp"
#include "gpu_opset/implementation_params.hpp"
#include "gpu_opset/implementation_registry.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/relu.hpp"

namespace ov {
namespace intel_gpu {

struct ActivationParams : public FactoryParameters {
    enum class Type {
        ReLU,
        Abs,
        Undef
    };
    Type type = Type::Undef;

    ActivationParams() = default;
    ActivationParams(const ov::op::v0::Abs* node) : type(Type::Abs) {}
    ActivationParams(const ov::op::v0::Relu* node) : type(Type::ReLU) {}
};

class SomeActivationImpl : public OpImplementation {
public:
    SomeActivationImpl() : OpImplementation("SomeActivationImpl") {}

    void initialize(const ActivationParams& params) {
        m_params = params;
    }

    void execute() override {
        std::cerr << "SomeActivationImpl::execute()!\n";
    }

    ActivationParams m_params;
};

struct ActivationImplementationsRegistry : public ImplementationsRegistry {
    ActivationImplementationsRegistry() {
        register_impl<SomeActivationImpl>();
    }
    static const ActivationImplementationsRegistry& instance() {
        static ActivationImplementationsRegistry instance;
        return instance;
    }
};

REGISTER_OP_WITH_CUSTOM_PARAMS_AND_REGISTRY(Abs_v0, ov::op::v0::Abs, ActivationParams, ActivationImplementationsRegistry);
REGISTER_OP_WITH_CUSTOM_PARAMS_AND_REGISTRY(Relu_v0, ov::op::v0::Relu, ActivationParams, ActivationImplementationsRegistry);

}  // namespace intel_gpu
}  // namespace ov
