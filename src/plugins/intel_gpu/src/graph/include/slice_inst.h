// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <intel_gpu/primitives/slice.hpp>
#include "primitive_inst.h"

namespace cldnn {

using slice_node = typed_program_node<slice>;

template <>
struct typed_program_node<slice> : public typed_program_node_base<slice> {
    using parent = typed_program_node_base<slice>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {1, 2, 3, 4}; }
};

using slice_node = typed_program_node<slice>;

template <>
class typed_primitive_inst<slice> : public typed_primitive_inst_base<slice> {
    using parent = typed_primitive_inst_base<slice>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(slice_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(slice_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(slice_node const& node);

    typed_primitive_inst(network& network, slice_node const& desc);
};

using slice_inst = typed_primitive_inst<slice>;

} // namespace cldnn
