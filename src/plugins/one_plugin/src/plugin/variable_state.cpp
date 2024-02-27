// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "remote_context.hpp"
// #include "common_utils.hpp"
#include "remote_tensor.hpp"
#include "variable_state.hpp"
// #include "intel_gpu/runtime/memory_caps.hpp"
// #include "intel_gpu/runtime/layout.hpp"
// #include "intel_gpu/runtime/debug_configuration.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {

VariableState::VariableState(const VariableStateInfo& info/* , RemoteContextImpl::Ptr context, std::shared_ptr<cldnn::ShapePredictor> shape_predictor */)
    : VariableStateBase{info.m_id/* , context */}
    // , m_layout(info.m_layout)
    , m_user_specified_type(info.m_user_specified_type)
    // , m_shape_predictor(shape_predictor)
    // , m_initial_layout(info.m_layout)
    {
    update_device_buffer();
}

void VariableState::reset() {
}

void VariableState::set_state(const ov::SoPtr<ov::ITensor>& state) {
}

void VariableState::update_device_buffer() {
}

ov::element::Type VariableState::get_user_specified_type() const {
    return m_user_specified_type;
}

ov::SoPtr<ov::ITensor> VariableState::get_state() const {
    return {};
}

}  // namespace intel_gpu
}  // namespace ov
