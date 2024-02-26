// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
namespace ov {

class BuildImplementations: public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::BuildImplementations");

    BuildImplementations() : ov::pass::ModelPass() {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}   // namespace ov
