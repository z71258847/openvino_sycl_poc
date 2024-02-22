// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

namespace ov {

struct ImplSelector {
    static std::shared_ptr<ImplSelector> default_cpu_selector();
    static std::shared_ptr<ImplSelector> default_gpu_selector();
};

struct GPUImplSelector : public ImplSelector {

};

struct CPUImplSelector : public ImplSelector {

};

}  // namespace ov
