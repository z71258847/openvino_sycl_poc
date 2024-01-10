// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu_op_extension.hpp"

namespace ov {
namespace intel_gpu {

GPUOpExtension::~GPUOpExtension() = default;

void GPUOpExtension::visit_attributes(AttributeVisitor& visitor) {
    bool intel_gpu_opset = true;
    visitor.on_attribute("intel_gpu_opset", intel_gpu_opset);
    // visitor.on_attribute("preferred_input_fmts", m_preferred_input_fmts);
    // visitor.on_attribute("preferred_output_fmts", m_preferred_output_fmts);
    // visitor.on_attribute("impl_type", m_impl_type);
    visitor.on_attribute("optimized", m_optimized);
    visitor.on_attribute("share_buffer", m_share_buffer);
    // visitor.on_attribute("fused_ops", m_fused_ops);
}

void GPUOpExtension::set_preferred_input_fmt(size_t idx, Format type) {
    auto impl = get_preferred_impl_type();
    auto& formats = m_preferred_input_fmts[impl];
    if (idx >= formats.size())
        formats.resize(idx+1, Format::any);

    formats.at(idx) = type;
}

void GPUOpExtension::set_preferred_output_fmt(size_t idx, Format type) {
    auto impl = get_preferred_impl_type();
    auto& formats = m_preferred_output_fmts[impl];
    if (idx >= formats.size())
        formats.resize(idx+1, Format::any);

    formats.at(idx) = type;
}

void GPUOpExtension::set_preferred_input_fmts(ImplTypes impl_type, std::vector<Format> fmts) {
    m_preferred_input_fmts[impl_type] = fmts;
}

void GPUOpExtension::set_preferred_output_fmts(ImplTypes impl_type, std::vector<Format> fmts) {
    m_preferred_output_fmts[impl_type] = fmts;
}

std::vector<Format> GPUOpExtension::get_preferred_input_fmts(ImplTypes impl_type) const {
    if (m_preferred_input_fmts.count(impl_type) > 0)
        return m_preferred_input_fmts.at(impl_type);
    return {};
}

std::vector<Format> GPUOpExtension::get_preferred_output_fmts(ImplTypes impl_type) const {
    if (m_preferred_output_fmts.count(impl_type) > 0)
        return m_preferred_output_fmts.at(impl_type);
    return {};
}

Format GPUOpExtension::get_preferred_input_fmt(ImplTypes impl_type, size_t idx) const {
    OPENVINO_ASSERT(m_preferred_input_fmts.count(impl_type) > 0, "[GPU] get_preferred_input_fmt");
    return (idx < m_preferred_input_fmts.at(impl_type).size()) ? m_preferred_input_fmts.at(impl_type).at(idx) : Format(Format::any);
}

Format GPUOpExtension::get_preferred_output_fmt(ImplTypes impl_type, size_t idx) const {
    OPENVINO_ASSERT(m_preferred_output_fmts.count(impl_type) > 0, "[GPU] get_preferred_output_fmt");
    return (idx < m_preferred_output_fmts.at(impl_type).size()) ? m_preferred_output_fmts.at(impl_type).at(idx) : Format(Format::any);
}

std::set<ImplTypes> GPUOpExtension::get_available_impl_types() const {
    return m_available_impl_types;
}

void GPUOpExtension::set_available_impl_types(const std::set<ImplTypes>& impls) {
    m_available_impl_types = impls;
}

std::vector<Format> GPUOpExtension::get_preferred_input_fmts() const {
    return get_preferred_input_fmts(get_preferred_impl_type());
}
std::vector<Format> GPUOpExtension::get_preferred_output_fmts() const {
    return get_preferred_output_fmts(get_preferred_impl_type());
}

std::vector<Format>& GPUOpExtension::get_preferred_input_fmts() {
    return get_preferred_input_fmts(get_preferred_impl_type());
}

std::vector<Format>& GPUOpExtension::get_preferred_output_fmts() {
    return get_preferred_output_fmts(get_preferred_impl_type());
}

std::vector<Format>& GPUOpExtension::get_preferred_input_fmts(ImplTypes impl_type) {
    const size_t inputs_count = 1; // get_input_size()
    if (m_preferred_input_fmts.count(impl_type) == 0)
        m_preferred_input_fmts[impl_type].resize(inputs_count, Format::any);

    return m_preferred_input_fmts.at(impl_type);
}

std::vector<Format>& GPUOpExtension::get_preferred_output_fmts(ImplTypes impl_type) {
    const size_t outputs_count = 1; // get_output_size()
    if (m_preferred_output_fmts.count(impl_type) == 0)
        m_preferred_output_fmts[impl_type].resize(outputs_count, Format::any);

    return m_preferred_output_fmts.at(impl_type);
}

Format GPUOpExtension::get_preferred_input_fmt(size_t idx) const {
    if (get_preferred_impl_type() == ImplTypes::any)
        return Format::any;
    return get_preferred_input_fmt(get_preferred_impl_type(), idx);
}

Format GPUOpExtension::get_preferred_output_fmt(size_t idx) const {
    if (get_preferred_impl_type() == ImplTypes::any)
        return Format::any;
    return get_preferred_output_fmt(get_preferred_impl_type(), idx);
}

void GPUOpExtension::copy_preferred_params(const GPUOpExtension& other) {
    m_preferred_input_fmts = other.m_preferred_input_fmts;
    m_preferred_output_fmts = other.m_preferred_output_fmts;
    m_impl_type = other.m_impl_type;
    m_available_impl_types = other.m_available_impl_types;
}

void GPUOpExtension::copy_preferred_output_fmts(const GPUOpExtension& other) {
    auto other_fmts = other.get_preferred_output_fmts();
    for (auto& kv : m_preferred_output_fmts) {
        auto current_fmts = kv.second;
        for (size_t i = 0; i < other_fmts.size(); i++) {
            if (current_fmts.size() <= i)
                break;
            if (other_fmts[i] != Format::any)
                current_fmts[i] = other_fmts[i];
        }
    }
}

void GPUOpExtension::copy_preferred_input_fmts(const GPUOpExtension& other) {
    m_preferred_input_fmts = other.m_preferred_input_fmts;
}

void GPUOpExtension::set_implementation(std::unique_ptr<cldnn::primitive_impl> impl) {
    m_selected_impl = std::move(impl);
}


}  // namespace intel_gpu
}  // namespace ov
