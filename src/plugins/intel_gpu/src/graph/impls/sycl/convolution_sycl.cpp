// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_inst.h"
#include "intel_gpu/primitives/reorder.hpp"
#include "ocl/ocl_event.hpp"
#include "ocl/sycl_engine.hpp"
#include "ocl/sycl_stream.hpp"
#include "openvino/core/type/element_type.hpp"
#include "primitive_sycl_base.h"
#include "implementation_map.hpp"

#include "impls/ocl/kernel_selector_helper.h"

#include "sycl/sycl.hpp"
#include "sycl/ext/oneapi/experimental/builtins.hpp"

#include "impls/sycl/esimd_conv.h"

#include <algorithm>
#include <chrono>
#include <memory>


#ifdef __SYCL_DEVICE_ONLY__
          #define CONSTANT __attribute__((opencl_constant))
#else
          #define CONSTANT
#endif

namespace cldnn {
namespace sycl {

template<typename AType, typename WType, typename DType, typename BType, typename SType>
::sycl::event run_conv_i8(::sycl::queue& queue, const AType* a, const WType* w, DType* dst, const BType* b, const SType* s, size_t width, size_t height) {
    ::sycl::event e;
    size_t oWidth = (width + 1) / 2;
    size_t oHeight = (height + 1) / 2;
    ::sycl::range<2> work_group_size{ 8 , 8 };
    ::sycl::range<2> work_intems{ (size_t)oWidth /4, (size_t)oHeight/2 };
    ::sycl::nd_range<2>  rangs(work_intems, work_group_size);
    e = queue.submit([&](auto& h){  
        h.parallel_for(rangs,[=](::sycl::nd_item<2> it)  SYCL_ESIMD_KERNEL  {
            convolution7x7_3_64_s2_int8((uint8_t*)a, (uint8_t*)w, (uint8_t*)b, (uint8_t*)s, (uint8_t*)dst, width, height, it);
        });
    });
    return e;
}

struct convolution_sycl : typed_primitive_sycl_impl<convolution> {
    using parent = typed_primitive_sycl_impl<convolution>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::sycl::convolution_sycl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<convolution_sycl>(*this);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& /* events */, typed_primitive_inst<convolution>& instance) override {
        auto& network = instance.get_network();
        const auto& desc = instance.get_typed_desc<convolution>();
        const bool print = true;

        auto& stream = downcast<ocl::sycl_stream>(network.get_stream());
        auto& engine = downcast<ocl::sycl_engine>(network.get_engine());
        ::sycl::context sycl_context = engine.get_sycl_context();
        ::sycl::queue& sycl_queue = stream.get_sycl_queue();

        const auto& params = instance.get_impl_params();
        auto out_shape = params->output_layouts[0].get_shape();

        auto output = instance.output_memory_ptr(0);
        auto weights = instance.weights_memory();
        auto bias = instance.bias_term() ? instance.bias_memory() : nullptr;
        auto w_zp = instance.weights_zero_points_term() ? instance.weights_zero_points_memory() : nullptr;
        auto comp = instance.compensation_term() ? instance.compensation_memory() : nullptr;

        std::vector<memory::ptr> inputs = { instance.input_memory_ptr(0) };
        std::vector<memory::ptr> fused_inputs;
        
        std::cout << "input mem count:" << instance.inputs_memory_count() << std::endl;
        if (instance.has_fused_primitives()){
            std::cout << "fused mem count:" << instance.get_fused_mem_count() << std::endl;
            fused_inputs = {instance.fused_memory(0)};
        }

        std::cout << "input layout # for sycl conv:" << params->input_layouts.size() << std::endl;
        std::cout << "input # for sycl conv:" << inputs.size() << std::endl;
	
        ov::element::Type_t in_t = params->input_layouts[0].data_type;
        ov::element::Type_t bias_t = params->input_layouts[2].data_type;
        ov::element::Type_t scale_t = params->input_layouts[3].data_type;
        ov::element::Type_t wei_t = params->weights_layout.value().data_type;
        ov::element::Type_t out_t = params->output_layouts[0].data_type;
        
        size_t N = params->input_layouts[0].get_shape()[0];
        size_t C = params->input_layouts[0].get_shape()[1];
        size_t H = params->input_layouts[0].get_shape()[2];
        size_t W = params->input_layouts[0].get_shape()[3];

        void* in = static_cast<void*>(inputs[0]->buffer_ptr());
        void* wei = static_cast<void*>(weights->buffer_ptr());
        void* out = static_cast<void*>(output->buffer_ptr());
        void* b = nullptr;
        void* s = nullptr;
        if (bias!=nullptr) b = static_cast<void*>(bias->buffer_ptr());
        if (instance.has_fused_primitives()) s = static_cast<void*>(fused_inputs[0]->buffer_ptr());

        if (print) {
            std::cerr << "NCHW: " << N << " " << C << " " << H << " " << W << std::endl;

            std::cerr << "in: " << params->input_layouts[0].to_short_string() << std::endl;
            std::cerr << "wei: " << params->weights_layout.value().to_short_string() << std::endl;
            std::cerr << "bias: " << params->input_layouts[2].to_short_string() << std::endl;
            std::cerr << "scale: " << params->input_layouts[3].to_short_string() << std::endl;
            std::cerr << "out: " << params->output_layouts[0].to_short_string() << std::endl;

            std::cerr << "in_t = " << in_t << std::endl;
            std::cerr << "wei_t = " << wei_t << std::endl;
            std::cerr << "bias_t = " << bias_t << std::endl;
            std::cerr << "scale_t = " << scale_t << std::endl;
            std::cerr << "out_t = " << out_t << std::endl;

            std::cerr << "in = " << in << std::endl;
            std::cerr << "wei = " << wei << std::endl;
            std::cerr << "out = " << out << std::endl;
            std::cerr << "bias = " << b << std::endl;
            std::cerr << "s = " << s << std::endl;
        }

        bool barrier = stream.get_queue_type() == QueueTypes::out_of_order;

        return to_ocl_event(stream, run_conv_i8(sycl_queue, in, wei, out, b, s, W, H));
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params) {
        auto source_weights_layout = impl_params.get_input_layout(1);
        auto target_weights_layout = source_weights_layout;
        // target_weights_layout.format = format::oiyx;
        target_weights_layout.format = format::iyxo;

        return std::make_shared<WeightsReorderParams>(source_weights_layout, target_weights_layout);
    }

    static std::unique_ptr<primitive_impl> create(const convolution_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        return cldnn::make_unique<convolution_sycl>(engine, config, get_weights_reorder(impl_params));
    }
};

namespace detail {

attach_convolution_sycl::attach_convolution_sycl() {
    std::vector<data_types> dt = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
    };
    std::vector<format::type> fmt = {
        format::bfyx,
        format::byxf,
    };
    implementation_map<convolution>::add(impl_types::sycl, shape_types::dynamic_shape, convolution_sycl::create, dt, fmt);
    implementation_map<convolution>::add(impl_types::sycl, shape_types::static_shape, convolution_sycl::create, dt, fmt);
}

}  // namespace detail
}  // namespace sycl
}  // namespace cldnn
