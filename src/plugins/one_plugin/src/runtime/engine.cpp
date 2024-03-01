// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "engine.hpp"
#include "openvino/core/except.hpp"

namespace ov {
std::shared_ptr<Engine> Engine::create(const Device::Ptr device) {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<Engine> Engine::create() {
    OPENVINO_NOT_IMPLEMENTED;
}

}  // namespace ov
