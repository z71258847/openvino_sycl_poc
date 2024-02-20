// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "op_implementation.hpp"


namespace ov {
namespace intel_gpu {

class ImplementationsRegistry {
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
template<typename NodeType>
class TypedImplementationsRegistry : public ImplementationsRegistry {
public:
    static TypedImplementationsRegistry<NodeType>& instance() {
        static TypedImplementationsRegistry<NodeType> instance;
        return instance;
    }


private:
    TypedImplementationsRegistry() {
        std::cerr << "TypedImplementationsRegistry Default Ctor\n";
    };
};

template<>
class TypedImplementationsRegistry<ov::op::v0::Parameter> : public ImplementationsRegistry {
public:
    static TypedImplementationsRegistry<ov::op::v0::Parameter>& instance() {
        static TypedImplementationsRegistry<ov::op::v0::Parameter> instance;
        return instance;
    }
private:
    TypedImplementationsRegistry() {
        register_impl<SomeNodeImpl>();
    }
};

template<>
class TypedImplementationsRegistry<ov::op::v0::Result> : public ImplementationsRegistry {
public:
    static TypedImplementationsRegistry<ov::op::v0::Result>& instance() {
        static TypedImplementationsRegistry<ov::op::v0::Result> instance;
        return instance;
    }
private:
    TypedImplementationsRegistry() {
        register_impl<SomeNodeImpl1>();
    }
};

}  // namespace op
}  // namespace ov
