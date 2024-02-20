// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_extension.hpp"

namespace ov {
namespace intel_gpu {

void NodeExtension::visit_attributes(AttributeVisitor& visitor) {}

const MemoryDescs& NodeExtension::get_memory_desc() const { return m_memory_desc; }
void NodeExtension::set_memory_desc(const Argument& arg, const MemoryDesc& desc) { m_memory_desc[arg] = desc; }
void NodeExtension::set_memory_descs(const MemoryDescs& descs) { m_memory_desc = descs; }

void NodeExtension::set_inplace() { m_opt_attributes->m_inplace = true; }
bool NodeExtension::is_inplace() const { return m_opt_attributes->m_inplace; }

void NodeExtension::select_preferred_formats() {
    OPENVINO_ASSERT(m_node != nullptr);
    std::cerr << "select format for: " << m_node->get_friendly_name() << " " << m_node->get_type_name() << std::endl;
}

const ov::Node* NodeExtension::get_node_ptr() const { return m_node; }
void NodeExtension::set_node_ptr(const ov::Node* ptr) { m_node = ptr; }

}  // namespace intel_gpu
}  // namespace ov
