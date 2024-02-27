// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_utils.hpp"
#include "openvino/core/except.hpp"
#include "remote_context.hpp"
#include "remote_tensor.hpp"
#include "plugin.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {

RemoteTensorImpl::RemoteTensorImpl(RemoteContextImpl::Ptr context,
                                   const ov::Shape& shape,
                                   const ov::element::Type& element_type)
    : m_context(context)
    , m_element_type(element_type)
    , m_shape(shape) {
    update_hash();
    allocate();
}

RemoteTensorImpl::~RemoteTensorImpl() {
    deallocate();
}

const ov::element::Type& RemoteTensorImpl::get_element_type() const {
    return m_element_type;
}

const ov::Shape& RemoteTensorImpl::get_shape() const {
    return m_shape;
}

void RemoteTensorImpl::update_strides() {
    if (m_element_type.bitwidth() < 8)
        return;
    auto& shape = get_shape();
    m_strides.clear();
    if (!shape.empty()) {
        m_strides.resize(shape.size());
        m_strides.back() = shape.back() == 0 ? 0 : m_element_type.size();
        std::copy(shape.rbegin(), shape.rend() - 1, m_strides.rbegin() + 1);
        std::partial_sum(m_strides.rbegin(), m_strides.rend(), m_strides.rbegin(), std::multiplies<size_t>());
    }
}

const ov::Strides& RemoteTensorImpl::get_strides() const {
    return m_strides;
}

const AnyMap& RemoteTensorImpl::get_properties() const {
    return m_properties;
}

 void RemoteTensorImpl::set_shape(ov::Shape shape) {
}

bool RemoteTensorImpl::deallocate() noexcept {
    OPENVINO_NOT_IMPLEMENTED;
}

bool RemoteTensorImpl::is_allocated() const noexcept {
    OPENVINO_NOT_IMPLEMENTED;
}

void RemoteTensorImpl::allocate() {
    OPENVINO_NOT_IMPLEMENTED;
}

const std::string& RemoteTensorImpl::get_device_name() const {
    return m_context->get_device_name();
}

bool RemoteTensorImpl::is_shared() const noexcept {
    return m_mem_type == TensorType::BT_BUF_SHARED ||
           m_mem_type == TensorType::BT_USM_SHARED ||
           m_mem_type == TensorType::BT_IMG_SHARED ||
           m_mem_type == TensorType::BT_SURF_SHARED ||
           m_mem_type == TensorType::BT_DX_BUF_SHARED;
}

bool RemoteTensorImpl::supports_caching() const {
    return is_shared();
}

void RemoteTensorImpl::update_hash() {
}

bool RemoteTensorImpl::is_surface() const noexcept {
    return m_mem_type == TensorType::BT_SURF_SHARED ||
           m_mem_type == TensorType::BT_IMG_SHARED;
}

std::shared_ptr<RemoteContextImpl> RemoteTensorImpl::get_context() const {
    return m_context;
}

void RemoteTensorImpl::update_properties() {
}

}  // namespace intel_gpu
}  // namespace ov
