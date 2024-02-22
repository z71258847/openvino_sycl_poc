// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "joint_impl/implementation_params.hpp"
#include "openvino/core/except.hpp"

namespace ov {

class OpImplementation {
public:
    OpImplementation(std::string impl_name = "") : m_impl_name(impl_name) {}
    virtual void execute() = 0; // should return event ?
    std::string get_implementation_name() const { return m_impl_name; }

    virtual std::shared_ptr<OpImplementation> clone() const {
        OPENVINO_NOT_IMPLEMENTED;
    }

private:
    std::string m_impl_name;
};

using ImplementationBuilder = std::function<std::shared_ptr<OpImplementation>(const FactoryParameters&)>;
using BuildersList = std::vector<ImplementationBuilder>;
using ImplementationsList = std::vector<std::shared_ptr<OpImplementation>>;

}  // namespace ov
