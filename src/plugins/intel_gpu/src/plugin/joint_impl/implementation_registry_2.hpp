// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "op_implementation.hpp"


namespace ov {

struct ImplementationsRegistry {
public:
    virtual ~ImplementationsRegistry() = default;
    ImplementationsList get_all_impls() const { return m_impls; }

protected:
    ImplementationsRegistry() { }
    template <typename ImplType, typename std::enable_if<std::is_base_of<OpImplementation, ImplType>::value, bool>::type = true>
    void register_impl() {
        m_impls.push_back(std::make_shared<ImplType>());
    }

    ImplementationsList m_impls;
};

}  // namespace ov
