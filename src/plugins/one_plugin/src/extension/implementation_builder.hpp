// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "extension/executor.hpp"
#include "extension/op_implementation.hpp"
#include "openvino/core/except.hpp"

namespace ov {

struct ImplementationBuilder {
    using Ptr = std::shared_ptr<ImplementationBuilder>;

    virtual void add_impl(OpImplementation::Ptr impl) {
        // push to list
    }

    virtual void build() {}
};

struct CPUImplementationBuilder  : public ImplementationBuilder{
    using Ptr = std::shared_ptr<CPUImplementationBuilder>;
};

struct ImplementationBuilders {
    ImplementationBuilders();
    std::map<OpImplementation::Type, ImplementationBuilder::Ptr> m_builders;
    void add_impl(OpImplementation::Ptr impl) {
        if (m_builders.find(impl->get_type()) == m_builders.end())
            return;

        std::cerr << "Add impl " << impl->get_name() << " to builder\n";
        m_builders[impl->get_type()]->add_impl(impl);
    }

    void build() {
        for (auto& builder : m_builders) {
            builder.second->build();
        }
    }

};

}  // namespace ov
