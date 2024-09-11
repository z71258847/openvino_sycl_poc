// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_inst.h"
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

#include "impls/sycl/esimd_gemm_q4_0.h"
#include "impls/sycl/lnl_gemv.h"

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

template <typename A, typename B>
struct AccumulatorType {
    using type = float;
};

template<> struct AccumulatorType<::sycl::half, ::sycl::half> {
    using type = ::sycl::half;
};

template<> struct AccumulatorType<::sycl::half, uint8_t> {
    using type = ::sycl::half;
};


template<> struct AccumulatorType<::sycl::half, int8_t> {
    using type = ::sycl::half;
};

template<typename AType, typename WType, typename ScaleType, typename DType>
::sycl::event run_fc_q4_0_fp16out(::sycl::queue& queue, const AType* a, const WType* w, const ScaleType* s, DType* dst,
                              size_t M, size_t N, size_t K) {
  ::sycl::event e;
  if (M == 1) // GEMV
  {
    // K=4096
    uint32_t ppg=16;
    if (K==11008) ppg=64;

    int groups = (N + 7) / 8;
    ::sycl::range<1> GlobalRangeCommonDim4096(groups * 64);
    ::sycl::range<1> LocalRangeCommonDim4096(64);
    ::sycl::nd_range<1> RangeCommonDim4096(GlobalRangeCommonDim4096, LocalRangeCommonDim4096);

    if (K == 4096) {
      e = queue.submit([&](handler& cgh) {
        cgh.parallel_for(
            RangeCommonDim4096, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
              matrixMulCommonDim4096Int4NoReshape(
                  (uint8_t*)w,
                  (uint8_t*)a,
                  (uint8_t*)dst,
                  (uint8_t*)s,
                  ndi);
            });
      });
    }
  } else // GEMM
  {
    int groupReduce2048H = (N + 15) / 16;
    int groupReduce2048V = 1;
    int localReduce2048H = 64; // internalPrecision == 0  (fp32), not 32
    int localReduce2048V = 1;
    ::sycl::range<2> GlobalRangeReduce2048(
        groupReduce2048H * localReduce2048H,
        groupReduce2048V * localReduce2048V);
    ::sycl::range<2> LocalRangeReduce2048(localReduce2048H, localReduce2048V);
    ::sycl::nd_range<2> RangeReduce2048(
        GlobalRangeReduce2048, LocalRangeReduce2048);

    int lastReduce = 0;
    if (K == 4096) {
      for (int ii = 0; ii < 2; ii++) {
        lastReduce = (ii == 1);
        e = queue.submit([&](handler& cgh) {
          cgh.parallel_for(
              RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                gemmReduce2048WeightsQ40InputFp16_ipex(
                    (uint8_t*)w,
                    (uint8_t*)a,
                    (uint8_t*)dst,
                    (uint8_t*)s,
                    K,
                    M ,
                    ii,
                    lastReduce,
                    ndi);
              });
        });
      }
    }
  }
  return e;
}

struct fully_connected_sycl : typed_primitive_sycl_impl<fully_connected> {
    using parent = typed_primitive_sycl_impl<fully_connected>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::sycl::fully_connected_sycl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<fully_connected_sycl>(*this);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& /* events */, typed_primitive_inst<fully_connected>& instance) override {
        auto& network = instance.get_network();
        const auto& desc = instance.get_typed_desc<fully_connected>();
        const bool print = false;

        auto start = std::chrono::high_resolution_clock::now();

        auto& stream = downcast<ocl::sycl_stream>(network.get_stream());
        auto& engine = downcast<ocl::sycl_engine>(network.get_engine());
        ::sycl::context sycl_context = engine.get_sycl_context();
        ::sycl::queue& sycl_queue = stream.get_sycl_queue();
        auto end = std::chrono::high_resolution_clock::now();

        if (print)
            std::cerr << "init time: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " us" << std::endl;

        const auto& params = instance.get_impl_params();
        auto out_shape = params->output_layouts[0].get_shape();

        auto output = instance.output_memory_ptr(0);
        auto weights = instance.weights_memory();
        auto bias = instance.bias_term() ? instance.bias_memory() : nullptr;

        std::vector<memory::ptr> inputs = { instance.input_memory_ptr(0) };
        size_t in_id = instance.bias_term() ? 3 : 2;
        if (!desc->decompression_scale.empty())
            inputs.push_back(instance.dep_memory_ptr(in_id++));

        if (!desc->decompression_zero_point.empty())
            inputs.push_back(instance.dep_memory_ptr(in_id));

        OPENVINO_ASSERT(!instance.bias_term() && !instance.get_node().has_fused_primitives());

        ov::element::Type_t in_t = params->input_layouts[0].data_type;
        ov::element::Type_t wei_t = params->weights_layout.value().data_type;
        ov::element::Type_t out_t = params->output_layouts[0].data_type;
        ov::element::Type_t ds_t = params->input_layouts[2].data_type;
        ov::element::Type_t dzp_t = inputs.size() == 3 ? params->input_layouts[3].data_type : ov::element::Type_t::undefined;

        OPENVINO_ASSERT(out_shape.size() == 3);
        size_t M = out_shape[1];
        size_t N = out_shape[2];
        size_t K = params->weights_layout.value().get_partial_shape()[1].get_length();
        size_t groups_num = params->input_layouts[2].get_shape()[1];
        size_t group_size = K / groups_num;

        void* in = static_cast<void*>(inputs[0]->buffer_ptr());
        void* wei = static_cast<void*>(weights->buffer_ptr());
        void* out = static_cast<void*>(output->buffer_ptr());
        void* ds = static_cast<void*>(inputs[1]->buffer_ptr());
        void* dzp = inputs.size() == 3 ? static_cast<void*>(inputs[2]->buffer_ptr()) : nullptr;


        if (print) {
            std::cerr << "in: " << params->input_layouts[0].to_short_string() << std::endl;
            std::cerr << "wei: " << params->weights_layout.value().to_short_string() << std::endl;
            std::cerr << "out: " << params->output_layouts[0].to_short_string() << std::endl;
            std::cerr << "scale: " << params->input_layouts[2].to_short_string() << std::endl;
            std::cerr << "zp: " << (params->input_layouts.size() == 4 ? params->input_layouts[3].to_short_string()  : "none") << std::endl;

            std::cerr << "M = " << M << std::endl;
            std::cerr << "N = " << N << std::endl;
            std::cerr << "K = " << K << std::endl;
            std::cerr << "groups_num = " << groups_num << std::endl;
            std::cerr << "group_size = " << group_size << std::endl;
            std::cerr << "in_t = " << in_t << std::endl;
            std::cerr << "wei_t = " << wei_t << std::endl;
            std::cerr << "out_t = " << out_t << std::endl;
            std::cerr << "ds_t = " << ds_t << std::endl;
            std::cerr << "dzp_t = " << dzp_t << std::endl;

            std::cerr << "in = " << in << std::endl;
            std::cerr << "wei = " << wei << std::endl;
            std::cerr << "out = " << out << std::endl;
            std::cerr << "ds = " << ds << std::endl;
            std::cerr << "dzp = " << dzp << std::endl;
        }

        OPENVINO_ASSERT(inputs.size() >= 2);

        auto dzp_scalar = desc->decompression_zero_point_scalar;

        bool barrier = stream.get_queue_type() == QueueTypes::out_of_order;

        if (out_t == ov::element::f16){
          // const ::sycl::half* in = static_cast<const ::sycl::half*>(inputs[0]->buffer_ptr());
          // const uint8_t* wei = static_cast<const uint8_t*>(weights->buffer_ptr());
          // ::sycl::half* out = static_cast<::sycl::half*>(output->buffer_ptr());
          // const ::sycl::half* ds = static_cast<const ::sycl::half*>(inputs[1]->buffer_ptr());
          return to_ocl_event(stream, run_fc_q4_0_fp16out(sycl_queue, in, wei, ds, out, M, N, K));
        }
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params) {
        auto source_weights_layout = impl_params.get_input_layout(1);
        auto target_weights_layout = source_weights_layout;
        target_weights_layout.format = format::oiyx;

        return std::make_shared<WeightsReorderParams>(source_weights_layout, target_weights_layout);
    }

    static std::unique_ptr<primitive_impl> create(const fully_connected_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        return cldnn::make_unique<fully_connected_sycl>(engine, config, get_weights_reorder(impl_params));
    }
};

namespace detail {

attach_fully_connected_sycl::attach_fully_connected_sycl() {
    std::vector<data_types> dt = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
    };
    std::vector<format::type> fmt = {
        format::bfyx,
    };
    implementation_map<fully_connected>::add(impl_types::sycl, shape_types::dynamic_shape, fully_connected_sycl::create, dt, fmt);
    implementation_map<fully_connected>::add(impl_types::sycl, shape_types::static_shape, fully_connected_sycl::create, dt, fmt);
}

}  // namespace detail
}  // namespace sycl
}  // namespace cldnn