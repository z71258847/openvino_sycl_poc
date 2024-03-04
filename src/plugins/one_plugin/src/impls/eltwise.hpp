// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extension/implementation_registry.hpp"
#include "extension/implementation_params.hpp"
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

    EltwiseParams(const ov::op::v1::Add* node) : ImplementationParameters(node), type(Type::Add) {}
    EltwiseParams(const ov::op::v1::Subtract* node) : ImplementationParameters(node), type(Type::Sub) {}
};

struct EltwiseRegistry : public ImplementationsRegistry {
    EltwiseRegistry();
    static const EltwiseRegistry& instance() {
        static EltwiseRegistry instance;
        return instance;
    }
};

}  // namespace ov
