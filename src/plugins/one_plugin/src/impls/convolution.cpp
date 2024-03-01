// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extension/extended_opset.hpp"
#include "extension/implementation_args.hpp"
#include "extension/implementation_params.hpp"
#include "extension/implementation_registry.hpp"
#include "extension/memory_descriptor.hpp"
#include "extension/node_extension.hpp"
#include "convolution.hpp"
#include "opset/placeholder.hpp"

#include "backend/cpu/convolution_cpu.hpp"

#include <memory>

namespace ov {

ConvolutionImplementationsRegistry::ConvolutionImplementationsRegistry() {
    register_impl<cpu::SomeConvolutionCPUImpl>();
}

template<>
class TypedNodeExtension<intel_gpu::op::Convolution> : public TypedNodeExtensionBase<intel_gpu::op::Convolution>, public intel_gpu::op::Convolution {
public:

    explicit TypedNodeExtension(std::shared_ptr<intel_gpu::op::Convolution> base_op)
        : TypedNodeExtensionBase<intel_gpu::op::Convolution>(), intel_gpu::op::Convolution(*base_op) {}
    MemoryDescs get_default_descriptors() const override {
        MemoryDescs descs;
        // Basic customization
        descs[Argument::input(0)] = MemoryDesc(Format::any, m_node->get_input_partial_shape(0));
        descs[Argument::weights()] = MemoryDesc(Format::any, m_node->get_input_partial_shape(1));
        descs[Argument::output(0)] = MemoryDesc(Format::any, m_node->get_output_partial_shape(0));

        // Check in/out node type
        if (ov::is_type<intel_gpu::op::Placeholder>(m_node->get_input_node_shared_ptr(2))) {
            descs[Argument::bias()] = MemoryDesc(Format::any, m_node->get_input_partial_shape(2));
        }

        // Check prev/next node layouts
        // if (auto ext_node = std::dynamic_pointer_cast<NodeExtension>(m_node->get_input_node_shared_ptr(0))) {
        //     if (ext_node->get_memory_desc().at(Argument::output(0)).m_format == Format::any)
        //         descs[Argument::input(0)] = MemoryDesc(Format::bfyx, m_node->get_input_partial_shape(0));
        // }

        return descs;
    }
};

REGISTER_IMPLS(Convolution, intel_gpu::op::Convolution, SomeCustomParams, ConvolutionImplementationsRegistry);

}  // namespace ov
