// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include "openvino/core/except.hpp"

namespace ov {

class OpExecutor {
public:
    using Ptr = std::shared_ptr<OpExecutor>;
    OpExecutor(std::string impl_name = "") : m_impl_name(impl_name) {}
    virtual void execute() = 0; // should return event ?
    std::string get_implementation_name() const { return m_impl_name; }

    virtual std::shared_ptr<OpExecutor> clone() const {
        OPENVINO_NOT_IMPLEMENTED;
    }

private:
    std::string m_impl_name;
};

}  // namespace ov
