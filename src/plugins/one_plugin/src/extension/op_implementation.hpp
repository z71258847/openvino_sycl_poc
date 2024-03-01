// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "extension/implementation_params.hpp"
#include "openvino/core/except.hpp"

namespace ov {

class OpExecutor;

enum class DeviceType {
    CPU = 0,
    GPU = 1,
    UNDEFINED = 128,
};

struct NodeAffinity {
    NodeAffinity() = default;
    explicit NodeAffinity(const DeviceType& type) : m_type(type), m_id(0) {}
    DeviceType m_type = DeviceType::UNDEFINED;
    size_t m_id = 0;
};

class OpImplementation {
public:
    enum class Type : uint8_t {
        CPU = 0,
        OCL = 1,
        SYCL = 2,
        REF = 3,
        UNKNOWN = 255,
    };
    using Ptr = std::shared_ptr<OpImplementation>;
    OpImplementation(std::string impl_name = "", Type type = Type::UNKNOWN) : m_impl_name(impl_name), m_type(type) {}
    std::string get_name() const { return m_impl_name; }

    virtual std::shared_ptr<OpExecutor> get_executor() const { OPENVINO_NOT_IMPLEMENTED; }
    virtual bool supports(const ImplementationParameters* params) const { return true; }
    virtual void initialize(const ImplementationParameters* params) { OPENVINO_NOT_IMPLEMENTED; }

    Type get_type() const {return m_type;}

private:
    std::string m_impl_name;
    Type m_type;
};

using ImplementationsList = std::vector<OpImplementation::Ptr>;

}  // namespace ov
