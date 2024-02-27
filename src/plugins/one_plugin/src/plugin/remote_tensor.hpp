// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef NOMINMAX
# define NOMINMAX
#endif

#ifdef _WIN32
# include <openvino/runtime/intel_gpu/ocl/dx.hpp>
#else
// # include <openvino/runtime/intel_gpu/ocl/va.hpp>
#endif
#include "openvino/runtime/iremote_tensor.hpp"

#include "common_utils.hpp"

#include <string>
#include <map>
#include <memory>

namespace ov {
namespace intel_gpu {
class RemoteContextImpl;

class RemoteTensorImpl : public ov::IRemoteTensor {
    friend class RemoteAllocator;
public:
    RemoteTensorImpl(std::shared_ptr<RemoteContextImpl> context,
                     const ov::Shape& shape,
                     const ov::element::Type& element_type);

    ~RemoteTensorImpl() override;
    const AnyMap& get_properties() const override;
    const std::string& get_device_name() const override;

    void set_shape(ov::Shape shape) override;
    const ov::element::Type& get_element_type() const override;
    const ov::Shape& get_shape() const override;
    const ov::Strides& get_strides() const override;

    void allocate();
    bool deallocate() noexcept;

    bool is_allocated() const noexcept;
    bool is_surface() const noexcept;
    bool is_shared() const noexcept;

    std::shared_ptr<RemoteContextImpl> get_context() const;

private:
    std::shared_ptr<RemoteContextImpl> m_context;

    ov::element::Type m_element_type;
    ov::Shape m_shape;
    ov::Strides m_strides{};
    ov::AnyMap m_properties;

    TensorType m_mem_type;

    uint32_t m_plane;
    size_t m_hash = 0;

    bool supports_caching() const;
    void update_hash();
    void update_strides();
    void update_properties();
};

}  // namespace intel_gpu
}  // namespace ov
