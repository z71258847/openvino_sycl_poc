// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "joint_impl/implementation_params.hpp"
#include "joint_impl/implementation_registry.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/relu.hpp"

namespace ov {

struct ActivationParams : public ImplementationParameters {
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

struct ActivationImplementationsRegistry : public ImplementationsRegistry<ActivationParams> {
    ActivationImplementationsRegistry();
    static const ActivationImplementationsRegistry& instance() {
        static ActivationImplementationsRegistry instance;
        return instance;
    }
};

}  // namespace ov
