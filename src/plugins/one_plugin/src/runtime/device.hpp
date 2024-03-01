// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "device_info.hpp"

#include <memory>

namespace ov {

const uint32_t INTEL_VENDOR_ID = 0x8086;

/// @brief Represents detected GPU device object. Use device_query to get list of available objects.
struct Device {
public:
    using Ptr = std::shared_ptr<Device>;
    virtual DeviceInfo get_info() const = 0;
    // virtual memory_capabilities get_mem_caps() const = 0;

    // virtual bool is_same(const device::ptr other) = 0;

    // float get_gops(ov::data_types dt) const;

    virtual ~Device() = default;
};

}  // namespace ov
