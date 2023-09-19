// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "input_layout_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <memory>
#include <algorithm>

namespace {
bool has_optimized_users(cldnn::input_layout_node const& node) {
    for (auto& user : node.get_users()) {
        if (user->can_be_optimized()) {
            return true;
        }
    }

    return false;
}
}  // namespace

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(input_layout)

input_layout_node::typed_program_node(const std::shared_ptr<input_layout> dprim, program& prog)
    : parent(dprim, prog) {
    can_share_buffer(false);
}

input_layout_inst::typed_primitive_inst(network& network, input_layout_node const& node)
    : parent(network, node, !node.is_dynamic() && (!network.is_internal() || has_optimized_users(node))) {
    _has_valid_input = false;  // by default input for 'input_layout' is invalid as long as user doesn't call set_data
}

event::ptr input_layout_inst::set_data(memory::ptr mem) {
    auto ol = get_node_output_layout();

    check_memory_to_set(*mem, ol);
    event::ptr ev = nullptr;
    const auto& engine = get_network().get_engine();
    auto& stream = get_network().get_stream();

    if (mem->is_allocated_by(engine)) {
        OPENVINO_ASSERT(!m_outputs.empty(), "[GPU] Can't set data for empty input memory");
        get_output(0).set_memory(mem);
        ev = stream.create_user_event(true);
    } else {
        if (!get_output(0).allocated()) {
            get_output(0).allocate(mem->get_layout(), engine.get_preferred_memory_allocation_type(), false);
        }

        if (ol.is_dynamic() && get_output(0).size() < mem->size()) {
            get_output(0).allocate(mem->get_layout(), engine.get_preferred_memory_allocation_type(), false);
        }
        mem_lock<uint8_t> src(mem, stream);
        ev = output_memory_ptr(0)->copy_from(stream, src.data(), false);
    }
    _has_valid_input = true;
    _output_changed = true;
    return ev;
}

void input_layout_inst::update_shape() {
    OPENVINO_ASSERT(!m_outputs.empty() && get_output(0).allocated(), "[GPU] input memory is not set");
    auto mem_layout = get_output(0).get_layout();
    if (_impl_params->get_output_layout() != mem_layout) {
        set_shape_change();
    }
    _impl_params->output_layouts[0] = mem_layout;
}

std::string input_layout_inst::to_string(input_layout_node const& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
