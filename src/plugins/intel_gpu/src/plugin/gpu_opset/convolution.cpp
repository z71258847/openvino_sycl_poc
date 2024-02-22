// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu_opset.hpp"
#include "gpu_opset/implementation_args.hpp"
#include "gpu_opset/implementation_params.hpp"
#include "gpu_opset/node_extension.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/op/convolution.hpp"
#include <memory>

namespace ov {
namespace intel_gpu {

using NodeType = ov::intel_gpu::op::Convolution;

class SomeConvolutionImpl : public OpImplementation {
public:
    SomeConvolutionImpl() : OpImplementation("SomeConvolutionImpl") {}
    void execute() override {
        std::cerr << "SomeConvolutionImpl::execute()!\n";
    }
};

class ConvolutionImplementationsRegistry : public ImplementationsRegistry {
public:
    ConvolutionImplementationsRegistry() {
        register_impl<SomeConvolutionImpl>();
    }
    static const ConvolutionImplementationsRegistry& instance() {
        static ConvolutionImplementationsRegistry instance;
        return instance;
    }
};

struct SomeCustomParams : FactoryParameters { };
class CustomFactory : public ImplementationsFactory {
public:
    CustomFactory(const ov::Node* node)
        : ImplementationsFactory(
            std::make_shared<TypedNodeParams<op::Convolution>>(dynamic_cast<const op::Convolution*>(node)),
            ConvolutionImplementationsRegistry::instance().get_all_impls()) {
        std::cerr << "CustomFactory impls factory for " << NodeType::get_type_info_static().name << std::endl;
        for (auto& impl : m_impls)
            std::cerr << impl->get_implementation_name() << std::endl;
    }

    bool supports(const FactoryParameters& params) const override {
        return supports_impl(static_cast<const SomeCustomParams&>(params));
    };

protected:
    bool supports_impl(const SomeCustomParams& params) const { return false; }
};

template<>
class TypedNodeExtension<op::Convolution> : public TypedNodeExtensionBase<op::Convolution> {
public:
    void select_preferred_formats() override {
        // Basic customization
        m_memory_desc[Argument::input(0)] = MemoryDesc(Format::any);
        m_memory_desc[Argument::weights()] = MemoryDesc(Format::any);
        m_memory_desc[Argument::output(0)] = MemoryDesc(Format::any);

        // Check in/out node type
        if (ov::is_type<op::Placeholder>(m_node->get_input_node_shared_ptr(2))) {
            m_memory_desc[Argument::bias()] = MemoryDesc(Format::any);
        }

        // Check prev/next node layouts
        if (auto ext_node = std::dynamic_pointer_cast<NodeExtension>(m_node->get_input_node_shared_ptr(0))) {
            if (ext_node->get_memory_desc().at(Argument::output(0)).m_format == Format::any)
                m_memory_desc[Argument::input(0)] = MemoryDesc(Format::bfyx);
        }
    }
};

REGISTER_OP_WITH_CUSTOM_FACTORY(Convolution_internal, ov::intel_gpu::op::Convolution, CustomFactory);

}  // namespace intel_gpu
}  // namespace ov
