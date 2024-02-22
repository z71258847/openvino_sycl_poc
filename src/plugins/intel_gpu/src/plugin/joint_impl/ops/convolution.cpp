// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "joint_impl/extended_opset.hpp"
#include "joint_impl/implementation_args.hpp"
#include "joint_impl/implementation_params.hpp"
#include "joint_impl/implementation_registry.hpp"
#include "joint_impl/node_extension.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/op/convolution.hpp"
#include <memory>

namespace ov {

using NodeType = ov::intel_gpu::op::Convolution;
struct SomeCustomParams : FactoryParameters {
    SomeCustomParams(const ov::intel_gpu::op::Convolution* node) : FactoryParameters(node) {}
};

class SomeConvolutionImpl : public OpImplementation {
public:
    SomeConvolutionImpl(const SomeCustomParams& params) : OpImplementation("SomeConvolutionImpl") {}
    void execute() override {
        std::cerr << "SomeConvolutionImpl::execute()!\n";
    }
};

class ConvolutionImplementationsRegistry : public ImplementationsRegistry<SomeCustomParams> {
public:
    ConvolutionImplementationsRegistry() {
        register_impl<SomeConvolutionImpl>();
    }
    static const ConvolutionImplementationsRegistry& instance() {
        static ConvolutionImplementationsRegistry instance;
        return instance;
    }
};


template<>
class TypedNodeExtension<intel_gpu::op::Convolution> : public TypedNodeExtensionBase<intel_gpu::op::Convolution> {
public:

    void select_preferred_formats() override {
        // Basic customization
        m_memory_desc[Argument::input(0)] = MemoryDesc(Format::any);
        m_memory_desc[Argument::weights()] = MemoryDesc(Format::any);
        m_memory_desc[Argument::output(0)] = MemoryDesc(Format::any);

        // Check in/out node type
        if (ov::is_type<intel_gpu::op::Placeholder>(m_node->get_input_node_shared_ptr(2))) {
            m_memory_desc[Argument::bias()] = MemoryDesc(Format::any);
        }

        // Check prev/next node layouts
        if (auto ext_node = std::dynamic_pointer_cast<NodeExtension>(m_node->get_input_node_shared_ptr(0))) {
            if (ext_node->get_memory_desc().at(Argument::output(0)).m_format == Format::any)
                m_memory_desc[Argument::input(0)] = MemoryDesc(Format::bfyx);
        }
    }
};

REGISTER_OP_1(Convolution, intel_gpu::op::Convolution, SomeCustomParams, ConvolutionImplementationsRegistry);

}  // namespace ov
