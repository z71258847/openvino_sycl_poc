// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <tuple>
#include <utility>

namespace ov {

enum class TensorType {
    BT_EMPTY,
    BT_BUF_INTERNAL,
    BT_BUF_SHARED,
    BT_USM_SHARED,
    BT_USM_HOST_INTERNAL,
    BT_USM_DEVICE_INTERNAL,
    BT_IMG_SHARED,
    BT_SURF_SHARED,
    BT_DX_BUF_SHARED,
};


}  // namespace ov
