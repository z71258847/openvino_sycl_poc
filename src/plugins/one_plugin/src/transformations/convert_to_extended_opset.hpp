// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
class ConvertToExtendedOpset: public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ConvertToExtendedOpset", "0");
    ConvertToExtendedOpset();
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}   // namespace ov
