// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/op.hpp"

#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/runtime/format.hpp"
#include "openvino/op/parameter.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "intel_gpu/graph/primitive_impl.hpp"

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace ov {
namespace intel_gpu {

using Format = cldnn::format;
using ImplTypes = cldnn::impl_types;

class OpImplementation {
public:
    OpImplementation(std::string impl_name = "") : m_impl_name(impl_name) {}
    virtual void execute() = 0;
    std::string get_implementation_name() const { return m_impl_name; }

private:
    std::string m_impl_name;
};
using ImplementationsList = std::vector<std::shared_ptr<OpImplementation>>;

class SomeNodeImpl : public OpImplementation {
public:
    SomeNodeImpl() : OpImplementation("SomeImpl") {}
    void execute() override {
        std::cerr << "SomeNodeImpl::execute()!\n";
    }
};

class SomeNodeImpl1 : public OpImplementation {
public:
    SomeNodeImpl1() : OpImplementation("SomeImpl1") {}
    void execute() override {
        std::cerr << "SomeNodeImpl1::execute()!\n";
    }
};

}  // namespace op
}  // namespace ov
