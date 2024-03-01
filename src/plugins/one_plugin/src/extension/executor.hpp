// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include "openvino/core/except.hpp"
#include "runtime/memory.hpp"
#include "runtime/stream.hpp"

namespace ov {

class OpExecutor {
public:
    using Ptr = std::shared_ptr<OpExecutor>;
    OpExecutor(std::string impl_name = "") : m_impl_name(impl_name) {}
    virtual Event::Ptr execute(Stream& stream, const MemoryArgs& args, const Events dep_events = {}) = 0;
    std::string get_implementation_name() const { return m_impl_name; }

    virtual std::shared_ptr<OpExecutor> clone() const {
        OPENVINO_NOT_IMPLEMENTED;
    }

private:
    std::string m_impl_name;
};

}  // namespace ov
