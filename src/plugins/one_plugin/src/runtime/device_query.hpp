// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "device.hpp"

#include <map>
#include <string>
#include <algorithm>

namespace ov {

// Fetches all available gpu devices with specific runtime and engine types and (optionally) user context/device handles
struct DeviceQuery {
public:
    static int device_id;
    explicit DeviceQuery();

    std::map<std::string, Device::Ptr> get_available_devices() const {
        return _available_devices;
    }

    ~DeviceQuery() = default;
private:
    std::map<std::string, Device::Ptr> _available_devices;
};
}  // namespace ov
