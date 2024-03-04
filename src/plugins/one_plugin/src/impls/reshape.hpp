// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extension/implementation_registry.hpp"
#include "extension/implementation_params.hpp"
#include "openvino/op/reshape.hpp"

namespace ov {

struct ReshapeParams : public ImplementationParameters {
    ReshapeParams(const ov::op::v1::Reshape* node) : ImplementationParameters(node) {}
};

struct ReshapeRegistry : public ImplementationsRegistry {
    ReshapeRegistry();
    static const ReshapeRegistry& instance() {
        static ReshapeRegistry instance;
        return instance;
    }
};

}  // namespace ov
