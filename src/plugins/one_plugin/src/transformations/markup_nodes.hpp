// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace intel_gpu {

class ShapeOfSubgraph : public RuntimeAttribute {
public:
    OPENVINO_RTTI("shape_of_subgraph", "0");

    ShapeOfSubgraph() = default;

    bool visit_attributes(AttributeVisitor& visitor) override {
        return true;
    }

    bool is_copyable() const override {
        return false;
    }
};

class RuntimeSkippable : public RuntimeAttribute {
public:
    OPENVINO_RTTI("runtime_skippable", "0");

    RuntimeSkippable() = default;

    bool visit_attributes(AttributeVisitor& visitor) override {
        return true;
    }

    bool is_copyable() const override {
        return false;
    }
};

class MarkupNodes: public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::intel_gpu::MarkupNodes");

    MarkupNodes() : ov::pass::ModelPass() {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}   // namespace intel_gpu
}   // namespace ov
