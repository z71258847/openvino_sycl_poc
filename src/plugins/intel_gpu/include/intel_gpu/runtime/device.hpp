// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "device_info.hpp"
#include "memory_caps.hpp"

#include <memory>

namespace cldnn {

/// @brief Represents detected GPU device object. Use device_query to get list of available objects.
struct device {
public:
    using ptr = std::shared_ptr<device>;
    virtual device_info get_info() const = 0;
    virtual memory_capabilities get_mem_caps() const = 0;

    virtual bool is_same(const device::ptr other) = 0;

    virtual ~device() = default;
};

struct dummy_device : public device {
public:
    dummy_device() :
        _mem_caps({allocation_type::cl_mem, allocation_type::usm_host, allocation_type::usm_device}) {
        device_info info;
        info.vendor_id = 0x0;
        info.dev_name = "Unknown";
        info.driver_version = "Unknown";
        info.dev_type = device_type::integrated_gpu;

        info.execution_units_count = 0;

        info.gpu_frequency = 0;

        info.max_work_group_size = 0;

        info.max_local_mem_size = 0;
        info.max_global_mem_size = 0;
        info.max_alloc_mem_size = 0;

        info.supports_image = false;
        info.max_image2d_width = 0;
        info.max_image2d_height = 0;

        // Check for supported featuresinfo.
        info.supports_fp16 = true;
        info.supports_fp64 = true;
        info.supports_fp16_denorms = true;

        info.supports_subgroups = true;
        info.supports_subgroups_short = true;
        info.supports_subgroups_char = true;

        info.supports_imad = true;
        info.supports_immad = false;

        info.supports_usm = true;

        info.supports_local_block_io = true;

        info.supports_queue_families = false;

        info.supported_simd_sizes = {8, 16, 32};
        info.gfx_ver = {0, 0, 0};
        info.device_id = 0;
        info.num_slices = 0;
        info.num_sub_slices_per_slice = 0;
        info.num_eus_per_sub_slice = 0;
        info.num_threads_per_eu = 0;

        _info = info;
    }

    device_info get_info() const override { return _info; }
    memory_capabilities get_mem_caps() const override { return _mem_caps; }

    bool is_same(const device::ptr other) override { return false; }

    ~dummy_device() = default;

private:
    device_info _info;
    memory_capabilities _mem_caps;
};

}  // namespace cldnn
