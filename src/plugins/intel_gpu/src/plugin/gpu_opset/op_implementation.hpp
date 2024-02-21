// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/runtime/format.hpp"
#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_gpu {

using Format = cldnn::format;
using ImplTypes = cldnn::impl_types;

class OpImplementation {
public:
    OpImplementation(std::string impl_name = "") : m_impl_name(impl_name) {}
    virtual void execute() = 0; // should return event ?
    std::string get_implementation_name() const { return m_impl_name; }

private:
    std::string m_impl_name;
};
using ImplementationsList = std::vector<std::shared_ptr<OpImplementation>>;

}  // namespace op
}  // namespace ov
