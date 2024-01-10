// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/execution_config.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/format.hpp"
#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

using DeviceInfo = cldnn::device_info;
using ImplTypes = cldnn::impl_types;
using Format = cldnn::format;
using cldnn::device_type;

namespace ov {
namespace intel_gpu {

class LayoutOptimizer final {
public:
    struct Attributes {
        bool use_onednn;
    };

    using PreferredFormats = std::pair<std::vector<Format>, std::vector<Format>>;

    LayoutOptimizer(const DeviceInfo& device_info, const ExecutionConfig& config, const Attributes& attrs);
    void select_preferred_formats(const std::shared_ptr<ov::Node>& node) const;
    ImplTypes get_preferred_impl_type(const std::shared_ptr<ov::Node>& node, Format preferred_format) const;
    ImplTypes get_preferred_impl_type(const std::shared_ptr<ov::Node>& node) const;
    PreferredFormats get_preferred_formats(const std::shared_ptr<ov::Node>& node, ImplTypes impl_type) const;
    bool is_optimized_format(Format fmt) const;
    bool is_format_supported(const ov::Node* op, Format fmt) const;

private:
    const DeviceInfo& m_device_info;
    const ExecutionConfig& m_config;
    Attributes m_attrs;
    ImplForcingMap m_forcing_map;
};

}   // namespace intel_gpu
}   // namespace ov
