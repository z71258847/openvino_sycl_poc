// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "joint_impl/implementation_registry.hpp"
#include "joint_impl/implementation_params.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {

struct EltwiseParams : public ImplementationParameters {
    enum class Type {
        Add,
        Sub,
        Undef
    };
    Type type = Type::Undef;

    EltwiseParams() = default;
    EltwiseParams(const ov::op::v1::Add* node) : type(Type::Add) {}
    EltwiseParams(const ov::op::v1::Subtract* node) : type(Type::Sub) {}
};

struct EltwiseRegistry : public ImplementationsRegistry {
    EltwiseRegistry();
    static const EltwiseRegistry& instance() {
        static EltwiseRegistry instance;
        return instance;
    }
};

}  // namespace ov
