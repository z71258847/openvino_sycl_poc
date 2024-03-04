// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_to_space.hpp"
#include "extension/implementation_params.hpp"
#include "extension/implementation_registry.hpp"
#include "openvino/op/batch_to_space.hpp"

namespace ov {

struct BatchToSpaceParams : public ImplementationParameters {
    explicit BatchToSpaceParams(const ov::op::v1::BatchToSpace* node) : ImplementationParameters(node), some_parameter(node->get_output_size()) {

    }
    int some_parameter = 100500;
};

struct BatchToSpaceImplementationsRegistry : public ImplementationsRegistry {
    BatchToSpaceImplementationsRegistry();
    static const BatchToSpaceImplementationsRegistry& instance() {
        static BatchToSpaceImplementationsRegistry instance;
        return instance;
    }
};

}  // namespace ov
