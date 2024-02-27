// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "implementation_factory.hpp"
#include "extension/implementation_selector.hpp"
#include "extension/node_extension.hpp"
#include "extension/op_implementation.hpp"
#include "openvino/core/node.hpp"


namespace ov {

void ImplementationsFactory::initialize_selector(const ov::Node* node) {
    m_impl_selector = dynamic_cast<const NodeExtension*>(node)->get_affinity().m_type == DeviceType::CPU ? ImplSelector::default_cpu_selector()
                                                                                                         : ImplSelector::default_gpu_selector();

}

}  // namespace ov
