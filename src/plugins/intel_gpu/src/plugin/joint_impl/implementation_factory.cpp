// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "implementation_factory.hpp"
#include "joint_impl/implementation_selector.hpp"
#include "joint_impl/node_extension.hpp"
#include "joint_impl/op_implementation.hpp"
#include "openvino/core/node.hpp"


namespace ov {

void ImplementationsFactory::initialize_selector(const ov::Node* node) {
    m_impl_selector = dynamic_cast<const NodeExtension*>(node)->get_affinity().m_type == DeviceType::CPU ? ImplSelector::default_cpu_selector()
                                                                                                         : ImplSelector::default_gpu_selector();

}

}  // namespace ov
