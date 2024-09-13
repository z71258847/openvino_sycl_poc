// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_inst.h"
#include "impls/registry/implementation_manager.hpp"

#include <memory>

namespace cldnn {
namespace sycl {

struct FCImplementationManagerSYCL : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("FCImplementationManagerSYCL")
    FCImplementationManagerSYCL(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::sycl, shape_type, vf) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<fully_connected>());

        static const std::vector<format::type> supported_formats = {
            format::bfyx,
        };

        const auto& fc_node = node.as<fully_connected>();
        const auto& in_layout = fc_node.get_input_layout(0);
        const auto& wei_layout = fc_node.weights().get_output_layout(false);
        const auto& out_layout = fc_node.get_output_layout(0);
        auto in0_dt = in_layout.data_type;
        auto wei_dt = wei_layout.data_type;
        auto out_dt = out_layout.data_type;
        auto fc_prim = fc_node.get_primitive();


        bool compressed_case = fc_prim->compressed_weights &&
                               one_of(in0_dt, {data_types::f16}) &&
                               one_of(wei_dt, {data_types::u4, data_types::i4}) &&
                               one_of(out_dt, {data_types::f16});
        if (!compressed_case)
            return false;

        if (wei_layout.get_partial_shape()[1] != 8192)
            return false;

        if (!one_of(in_layout.format.value, supported_formats) || !one_of(out_layout.format.value, supported_formats))
            return false;

        if (in_layout.data_padding || out_layout.data_padding)
            return false;

        std::cout << fc_node.id() << std::endl;
        std::cout << "use sycl impl" << std::endl;
        std::cout << wei_layout << std::endl;
        return true;
    }

    bool support_shapes(const kernel_impl_params& param) const override{
        // auto out_shape = param.output_layouts[0].get_shape();
        // std::cout << "out shape: " << out_shape << std::endl;
        // if (out_shape[0]>1) return false;

        return true;
    }
};

}  // namespace sycl
}  // namespace cldnn
