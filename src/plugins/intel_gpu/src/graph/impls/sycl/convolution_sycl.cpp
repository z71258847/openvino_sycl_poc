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

template<typename AType, typename WType, typename DType>
::sycl::event run_conv_i8(::sycl::queue& queue, const AType* a, const WType* w, DType* dst) {
    ::sycl::event e;
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
        auto a_zp = instance.activations_zero_points_term() ? instance.activations_zero_points_memory() : nullptr;

        std::vector<memory::ptr> inputs = { instance.input_memory_ptr(0) };
        size_t in_id = instance.bias_term() ? 3 : 2;

        std::cout << "input # for sycl conv:" << inputs.size() << std::endl;
	
        ov::element::Type_t in_t = params->input_layouts[0].data_type;
        ov::element::Type_t wei_t = params->weights_layout.value().data_type;
        ov::element::Type_t out_t = params->output_layouts[0].data_type;

        void* in = static_cast<void*>(inputs[0]->buffer_ptr());
        void* wei = static_cast<void*>(weights->buffer_ptr());
        void* out = static_cast<void*>(output->buffer_ptr());

        if (print) {
            std::cerr << "in: " << params->input_layouts[0].to_short_string() << std::endl;
            std::cerr << "wei: " << params->weights_layout.value().to_short_string() << std::endl;
            std::cerr << "out: " << params->output_layouts[0].to_short_string() << std::endl;

            std::cerr << "in_t = " << in_t << std::endl;
            std::cerr << "wei_t = " << wei_t << std::endl;
            std::cerr << "out_t = " << out_t << std::endl;

            std::cerr << "in = " << in << std::endl;
            std::cerr << "wei = " << wei << std::endl;
            std::cerr << "out = " << out << std::endl;
        }

        bool barrier = stream.get_queue_type() == QueueTypes::out_of_order;

        return to_ocl_event(stream, run_conv_i8(sycl_queue, in, wei, out));
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params) {
        auto source_weights_layout = impl_params.get_input_layout(1);
        auto target_weights_layout = source_weights_layout;
        // target_weights_layout.format = format::oiyx;
        target_weights_layout.format = format::ioyx;

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
    };
    implementation_map<convolution>::add(impl_types::sycl, shape_types::dynamic_shape, convolution_sycl::create, dt, fmt);
    implementation_map<convolution>::add(impl_types::sycl, shape_types::static_shape, convolution_sycl::create, dt, fmt);
}

}  // namespace detail
}  // namespace sycl
}  // namespace cldnn
