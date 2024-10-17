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
        bool fp16_case = one_of(in0_dt, {data_types::f16}) &&
                        one_of(wei_dt, {data_types::f16}) &&
                        one_of(out_dt, {data_types::f16});

        // std::string fc_node_model_id = fc_node.id().substr(0, 34);
        // if (fc_node_model_id == "fullyconnected:__module.text_model")
        // {
            
        //     std::cout << " ============================== skip text model: " << fc_node.id() << std::endl; 
        //     return false;
        // }
        
        if ((wei_layout.get_partial_shape()[1]!=1280 &&
            wei_layout.get_partial_shape()[1]!=2560 &&
            wei_layout.get_partial_shape()[1]!=2048 &&
            wei_layout.get_partial_shape()[1]!=640 &&
            wei_layout.get_partial_shape()[1]!=5120
            )
        ){
            return false;
        }
        if (!fp16_case){
            return false;
        }
        if (!one_of(in_layout.format.value, supported_formats) || !one_of(out_layout.format.value, supported_formats)){
            // std::cout << in_layout.format.value << std::endl; 
            // std::cout << out_layout.format.value << std::endl; 
            return false;
        }
        if (in_layout.data_padding || out_layout.data_padding){
            return false;
        }

        if (fc_node.has_fused_primitives()){
            // std::cout << "fused? " << fc_node.id() << std::endl;
            // return false;
        }

        std::cout << fc_node.id() << " possible use sycl!!!!!!!!!!!!!!" << std::endl;
        
        return true;
        // return false;
    }

    bool support_shapes(const kernel_impl_params& params) const override {
        if ((
            params.get_input_layout().get_partial_shape()[1]==77 || 
             params.get_input_layout().get_partial_shape()[1]==256 || 
             params.get_input_layout().get_partial_shape()[1]==1024)
         && 
            (params.get_output_layout().get_partial_shape()[2]==1280 ||
            params.get_output_layout().get_partial_shape()[2]==640 ||
            params.get_output_layout().get_partial_shape()[2]==5120 ||
            params.get_output_layout().get_partial_shape()[2]==10240
            )
            ){
            return true;
        }
        return false;
    }
};
}  // namespace sycl
}  // namespace cldnn