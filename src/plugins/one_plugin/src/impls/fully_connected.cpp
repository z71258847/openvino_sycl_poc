// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected.hpp"

#include "extension/extended_opset.hpp"
#include "backend/cpu/fully_connected_cpu.hpp"

namespace ov {

FullyConnectedImplementationsRegistry::FullyConnectedImplementationsRegistry() {
    register_impl<cpu::SomeFullyConnectedCPUImpl>();
}

template<>
class TypedNodeExtension<intel_gpu::op::FullyConnected> : public TypedNodeExtensionBase<intel_gpu::op::FullyConnected>, public intel_gpu::op::FullyConnected {
public:

    explicit TypedNodeExtension(std::shared_ptr<intel_gpu::op::FullyConnected> base_op)
        : TypedNodeExtensionBase<intel_gpu::op::FullyConnected>(), intel_gpu::op::FullyConnected(*base_op) {}
    MemoryDescs get_default_descriptors() const override {
        MemoryDescs descs;
        // Basic customization
        descs[Argument::input(0)] = MemoryDesc(m_node->get_input_element_type(0), m_node->get_input_partial_shape(0), Format::any);
        descs[Argument::weights()] = MemoryDesc(m_node->get_input_element_type(1), m_node->get_input_partial_shape(1), Format::any);
        descs[Argument::output(0)] = MemoryDesc(m_node->get_output_element_type(0), m_node->get_output_partial_shape(0), Format::any);

        // Check prev/next node layouts
        // if (auto ext_node = std::dynamic_pointer_cast<NodeExtension>(m_node->get_input_node_shared_ptr(0))) {
        //     if (ext_node->get_memory_desc().at(Argument::output(0)).m_format == Format::any)
        //         descs[Argument::input(0)] = MemoryDesc(Format::bfyx, m_node->get_input_partial_shape(0));
        // }

        return descs;
    }
};

REGISTER_IMPLS(FullyConnected, ov::intel_gpu::op::FullyConnected, FullyConnectedParams, FullyConnectedImplementationsRegistry);

}  // namespace ov
