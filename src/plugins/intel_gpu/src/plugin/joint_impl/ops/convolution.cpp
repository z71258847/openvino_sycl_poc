// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "joint_impl/extended_opset.hpp"
#include "joint_impl/implementation_args.hpp"
#include "joint_impl/implementation_params.hpp"
#include "joint_impl/implementation_registry.hpp"
#include "joint_impl/node_extension.hpp"
#include "joint_impl/ops/convolution.hpp"
#include "intel_gpu/op/placeholder.hpp"

#include "joint_impl/ops/cpu/convolution_cpu.hpp"

#include <memory>

namespace ov {

ConvolutionImplementationsRegistry::ConvolutionImplementationsRegistry() {
    register_impl<cpu::SomeConvolutionCPUImpl>();
}

template<>
class TypedNodeExtension<intel_gpu::op::Convolution> : public TypedNodeExtensionBase<intel_gpu::op::Convolution> {
public:

    void initialize_descriptors() override {
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

REGISTER_IMPLS(Convolution, intel_gpu::op::Convolution, SomeCustomParams, ConvolutionImplementationsRegistry);

}  // namespace ov
