// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "joint_impl/executor.hpp"
#include "joint_impl/implementation_params.hpp"
#include "openvino/core/except.hpp"

namespace ov {

class OpImplementation {
public:
    using Ptr = std::shared_ptr<OpImplementation>;
    OpImplementation(std::string impl_name = "") : m_impl_name(impl_name) {}
    std::string get_implementation_name() const { return m_impl_name; }

    virtual std::shared_ptr<OpExecutor> get_executor(const ImplementationParameters* params) const { OPENVINO_NOT_IMPLEMENTED; }
    virtual bool supports(const ImplementationParameters* params) const { return true; }

private:
    std::string m_impl_name;
};

using ImplementationsList = std::vector<OpImplementation::Ptr>;

}  // namespace ov
