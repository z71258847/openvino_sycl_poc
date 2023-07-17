// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <slice_inst.h>
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>

#include "slice_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(slice)

slice_inst::typed_primitive_inst(network& network, slice_node const& node)
    : parent(network, node) {}

layout slice_inst::calc_output_layout(slice_node const& node, kernel_impl_params const& impl_param) {
    auto primitive = impl_param.typed_desc<slice>();
    auto input_layout = impl_param.get_input_layout();
    return {input_layout.data_type, input_layout.format, primitive->output_shape};
}

template<typename ShapeType>
std::vector<layout> slice_inst::calc_output_layouts(slice_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<slice>();
    auto input_layout = impl_param.get_input_layout(0);
    auto start_layout = impl_param.get_input_layout(1);
    auto stop_layout = impl_param.get_input_layout(2);
    auto step_layout = impl_param.get_input_layout(3);
    auto axis_layout = impl_param.get_input_layout(4);

    auto& constant_mem = impl_param.memory_deps;
    ov::op::v8::Slice op;

    std::vector<ShapeType> output_shapes = {ShapeType{}};
    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>(),
        start_layout.get<ShapeType>(),
        stop_layout.get<ShapeType>(),
        step_layout.get<ShapeType>(),
        axis_layout.get<ShapeType>(),
    };

    if (constant_mem.size() == 4) {
        std::map<size_t, ngraph::HostTensorPtr> const_data;
        auto start_mem = constant_mem.at(1);
        auto stop_mem = constant_mem.at(2);
        auto step_mem = constant_mem.at(3);
        auto axis_mem = constant_mem.at(4);

        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock1(start_mem, impl_param.get_stream());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock2(stop_mem, impl_param.get_stream());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock3(step_mem, impl_param.get_stream());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock4(axis_mem, impl_param.get_stream());

        auto start_tensor = make_host_tensor(start_mem->get_layout(), lock1.data());
        auto stop_tensor = make_host_tensor(stop_mem->get_layout(), lock2.data());
        auto step_tensor = make_host_tensor(step_mem->get_layout(), lock3.data());
        auto axis_tensor = make_host_tensor(axis_mem->get_layout(), lock4.data());

        const_data.emplace(1, start_tensor);
        const_data.emplace(2, stop_tensor);
        const_data.emplace(3, step_tensor);
        const_data.emplace(4, axis_tensor);

        ov::op::v8::shape_infer(&op, input_shapes, output_shapes, const_data);

        auto output_format = format::get_default_format(output_shapes[0].size());
        return { layout{output_shapes[0], input_layout.data_type, output_format} };
    } else {
        auto out_shape = ov::PartialShape::dynamic(input_layout.get_partial_shape().size());
        return { layout{out_shape, input_layout.data_type, format::get_default_format(out_shape.rank().get_length())} };
    }
}

template std::vector<layout> slice_inst::calc_output_layouts<ov::PartialShape>(slice_node const& node, const kernel_impl_params& impl_param);

std::string slice_inst::to_string(slice_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite slice_info;
    slice_info.add("input id", node.input().id());
    slice_info.add("begin_param id", node.get_dependency(1).id());
    slice_info.add("end_param id", node.get_dependency(2).id());
    slice_info.add("step_param id", node.get_dependency(3).id());
    slice_info.add("axis_param id", node.get_dependency(4).id());
    node_info->add("slice info", slice_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

} // namespace cldnn
