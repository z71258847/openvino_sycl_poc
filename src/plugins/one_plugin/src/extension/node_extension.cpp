// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_extension.hpp"
#include "extension/implementation_args.hpp"
#include "extension/op_implementation.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"


namespace ov {

void NodeExtension::visit_attributes(AttributeVisitor& visitor) {}

const Configuration& NodeExtension::get_best_configuration() const { return m_best_config; }
void NodeExtension::set_best_configuration(const Configuration& best_config) { m_best_config = best_config; }
const std::vector<Configuration>& NodeExtension::get_available_configurations() const { return m_available_configs; }

void NodeExtension::set_inplace() { m_opt_attributes->m_inplace = true; }
bool NodeExtension::is_inplace() const { return m_opt_attributes->m_inplace; }

std::shared_ptr<OpImplementation> NodeExtension::get_impl() const {
    return m_best_implementation;
}

MemoryDescs NodeExtension::get_default_descriptors() const {
    MemoryDescs desc;
    for (size_t i = 0; i < m_node->get_input_size(); i++) {
        desc[Argument::input(i)] = MemoryDesc(Format::any, m_node->get_input_partial_shape(i));
    }

    for (size_t i = 0; i < m_node->get_output_size(); i++) {
        desc[Argument::output(i)] = MemoryDesc(Format::any, m_node->get_output_partial_shape(i));
    }

    if (m_fused_ops) {
        size_t i = 0;
        for (auto& op : m_fused_ops->get_ordered_ops()) {
            if (ov::is_type<op::v0::Parameter>(op)) {
                desc[Argument::post_op(i++)] = MemoryDesc(Format::any, op->get_output_partial_shape(0));
            }
        }
    }

    return desc;
}

void NodeExtension::select_preferred_formats(std::shared_ptr<const LayoutOptimizer> layout_optimizer) {
    OPENVINO_ASSERT(m_node != nullptr);
    std::cerr << "select format for: " << m_node->get_friendly_name() << " " << m_node->get_type_name() << std::endl;
    m_available_configs = layout_optimizer->get_available_configurations(this);
}

const ov::Node* NodeExtension::get_node_ptr() const { return m_node; }

std::shared_ptr<OpExecutor> NodeExtension::get_executor() const {
    return m_executor;
}

void NodeExtension::create_executor(const ImplementationBuilders& builder) {
    m_executor = m_factory->create_executor(m_best_implementation, *builder.m_builders.at(m_best_implementation->get_type()));
}

void NodeExtension::add_fused_op(std::shared_ptr<ov::Node> op) {
    // somehow modify m_fused_ops and add op to it
}

void NodeExtension::set_fused_ops(std::shared_ptr<ov::Model> fused_ops) {
    m_fused_ops = fused_ops;
}

void NodeExtension::set_affinity(const NodeAffinity& affinity) {
    m_affinity = affinity;
}

void NodeExtension::set_affinity(const DeviceType& device_type) {
    m_affinity = NodeAffinity{device_type};
}

NodeAffinity NodeExtension::get_affinity() const {
    return m_affinity;
}

}  // namespace ov
