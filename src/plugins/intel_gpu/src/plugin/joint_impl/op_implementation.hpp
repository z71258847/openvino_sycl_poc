// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ov {

class OpImplementation {
public:
    OpImplementation(std::string impl_name = "") : m_impl_name(impl_name) {}
    virtual void execute() = 0; // should return event ?
    std::string get_implementation_name() const { return m_impl_name; }

private:
    std::string m_impl_name;
};
using ImplementationsList = std::vector<std::shared_ptr<OpImplementation>>;

}  // namespace ov
