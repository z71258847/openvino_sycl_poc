// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class ConvertToGpuOpset: public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ConvertToGpuOpset", "0");
    ConvertToGpuOpset();
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}   // namespace intel_gpu
}   // namespace ov
