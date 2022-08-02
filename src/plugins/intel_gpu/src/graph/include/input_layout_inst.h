// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/input_layout.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {
struct memory;

template <>
struct typed_program_node<input_layout> : public typed_program_node_base<input_layout> {
    using parent = typed_program_node_base<input_layout>;
    using parent::parent;

    typed_program_node(const std::shared_ptr<input_layout> prim, program& prog);

    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params() const override {
        return parent::get_kernel_impl_params({}, get_primitive()->layout);
    }
};

using input_layout_node = typed_program_node<input_layout>;

template <>
class typed_primitive_inst<input_layout> : public typed_primitive_inst_base<input_layout> {
    using parent = typed_primitive_inst_base<input_layout>;

public:
    static layout calc_output_layout(input_layout_node const& node, kernel_impl_params const& impl_param) {
        return impl_param.output_layout;
    }

    void update_shape() override {
        if (!_output)
            OPENVINO_ASSERT(false, "[GPU] Can't update shape for input_layout instance as memory is not set");
        _impl_params->output_layout = _output->get_layout();
    }
    static std::string to_string(input_layout_node const& node);

    typed_primitive_inst(network& network, input_layout_node const& node);

    void set_data(memory::ptr mem);
};

using input_layout_inst = typed_primitive_inst<input_layout>;

}  // namespace cldnn
