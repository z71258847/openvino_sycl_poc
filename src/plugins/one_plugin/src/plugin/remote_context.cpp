// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/except.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "remote_context.hpp"
#include "remote_tensor.hpp"
#include <memory>

namespace ov {
namespace intel_gpu {

namespace {

template <typename Type>
Type extract_object(const ov::AnyMap& params, const ov::Property<Type>& p) {
    auto itrHandle = params.find(p.name());
    OPENVINO_ASSERT(itrHandle != params.end(), "[GPU] No parameter ", p.name(), " found in parameters map");
    ov::Any res = itrHandle->second;
    return res.as<Type>();
}

}  // namespace

RemoteContextImpl::RemoteContextImpl(const std::string& device_name/* , std::vector<cldnn::device::ptr> devices */) /* : m_device_name(device_name) */ {
}

RemoteContextImpl::RemoteContextImpl(const std::map<std::string, RemoteContextImpl::Ptr>& known_contexts, const AnyMap& params) {
}

void RemoteContextImpl::init_properties() {
}

const ov::AnyMap& RemoteContextImpl::get_property() const {
    return properties;
}

std::shared_ptr<RemoteContextImpl> RemoteContextImpl::get_this_shared_ptr() {
    return std::static_pointer_cast<RemoteContextImpl>(shared_from_this());
}

ov::SoPtr<ov::ITensor> RemoteContextImpl::create_host_tensor(const ov::element::Type type, const ov::Shape& shape) {
    return { ov::make_tensor(type, shape), nullptr };
}

ov::SoPtr<ov::IRemoteTensor> RemoteContextImpl::create_tensor(const ov::element::Type& type, const ov::Shape& shape, const ov::AnyMap& params) {
    OPENVINO_NOT_IMPLEMENTED;
}

const std::string& RemoteContextImpl::get_device_name() const {
    static std::string name = "ONE";
    return name;
}

}  // namespace intel_gpu
}  // namespace ov
