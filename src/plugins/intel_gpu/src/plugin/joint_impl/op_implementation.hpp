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

enum class DeviceType {
    CPU = 0,
    GPU = 1,
    UNDEFINED = 128,
};

struct NodeAffinity {
    NodeAffinity() = default;
    explicit NodeAffinity(const DeviceType& type) : m_type(type), m_id(0) {}
    DeviceType m_type = DeviceType::UNDEFINED;
    size_t m_id = 0;
};

class OpImplementation {
public:
    enum class Type {
        CPU = 0,
        OCL = 1,
        UNKNOWN = 1,
    };
    using Ptr = std::shared_ptr<OpImplementation>;
    OpImplementation(std::string impl_name = "", Type type = Type::UNKNOWN) : m_impl_name(impl_name), m_type(type) {}
    std::string get_implementation_name() const { return m_impl_name; }

    virtual std::shared_ptr<OpExecutor> get_executor(const ImplementationParameters* params) const { OPENVINO_NOT_IMPLEMENTED; }
    virtual bool supports(const ImplementationParameters* params) const { return true; }

    Type get_type() const {return m_type;}

private:
    std::string m_impl_name;
    Type m_type;
};

using ImplementationsList = std::vector<OpImplementation::Ptr>;


struct ImplementationBuilder {
    using Ptr = std::shared_ptr<ImplementationBuilder>;

    virtual void add_impl(OpImplementation::Ptr impl) {
        // push to list
    }
};

struct OCLImplementationBuilder : public ImplementationBuilder {
    using Ptr = std::shared_ptr<OCLImplementationBuilder>;
};

struct CPUImplementationBuilder  : public ImplementationBuilder{
    using Ptr = std::shared_ptr<CPUImplementationBuilder>;
};

struct ImplementationBuilders {
    std::map<OpImplementation::Type, ImplementationBuilder::Ptr> m_builders = {
        { OpImplementation::Type::OCL, std::make_shared<OCLImplementationBuilder>() }
    };

    void add_impl(OpImplementation::Ptr impl) {
        m_builders[OpImplementation::Type::OCL]->add_impl(impl);
    }

    void build() {

    }

};

}  // namespace ov
