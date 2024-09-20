// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_sycl.hpp"
#include "fully_connected_inst.h"
#include "intel_gpu/primitives/reorder.hpp"
#include "ocl/ocl_event.hpp"
#include "ocl/sycl_engine.hpp"
#include "ocl/sycl_stream.hpp"
#include "openvino/core/type/element_type.hpp"
#include "primitive_sycl_base.h"

#include "impls/ocl/kernel_selector_helper.h"

#include "sycl/sycl.hpp"
#include "sycl/ext/oneapi/experimental/builtins.hpp"

#include "impls/sycl/esimd_gemm_q4_0.h"
#include "impls/sycl/esimd_gemv_q4_0.h"

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
                              size_t M, size_t N, size_t K, size_t group_size, size_t groups_num) {
    ::sycl::event e;
    // if (M==1){ // GEMV
    if (K==8192 && N==2048){
        uint32_t groupsV2 = (N + 32 - 1) / 32;
        ::sycl::range<1> GlobalRangeCommonDim8192(groupsV2 * 8);
        ::sycl::range<1> LocalRangeCommonDim8192(8);
        ::sycl::nd_range<1> RangeCommonDim8192(
            GlobalRangeCommonDim8192, LocalRangeCommonDim8192);

        e = queue.submit([&](handler& cgh) {
        cgh.parallel_for(
            RangeCommonDim8192, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                GEMV_Int4Weight_FP16InOutNx16Temp_largeGRF_block_8T<8192, 5>(
                    (uint8_t*)w,
                    (uint8_t*)a,
                    (uint8_t*)dst,
                    (uint8_t*)s,
                    N,
                    ndi);
            });
        });
    }
    else if (K==3072 && N==128256){
        uint32_t groupsV2 = (N + 4 - 1) / 4;
        ::sycl::range<1> GlobalRangeCommonDim3072(groupsV2 * 6);
        ::sycl::range<1> LocalRangeCommonDim3072(6);
        ::sycl::nd_range<1> RangeCommonDim3072(
            GlobalRangeCommonDim3072, LocalRangeCommonDim3072);
  
        e = queue.submit([&](handler& cgh) {
          cgh.parallel_for(
              RangeCommonDim3072, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                GEMV_Int4Weight_FP16InOutNx16Temp_largeGRF_block_ppg8_8T<3072, 2>(
                    (uint8_t*)w,
                    (uint8_t*)a,
                    (uint8_t*)dst,
                    (uint8_t*)s,
                    N,
                    ndi);
              });
        });
    }
    else if (K==2048 && N==128256){
        uint32_t groupsV2 = (N + 8 - 1) / 8;
        ::sycl::range<1> GlobalRangeCommonDim2048(groupsV2 * 4);
        ::sycl::range<1> LocalRangeCommonDim2048(4);
        ::sycl::nd_range<1> RangeCommonDim2048(
            GlobalRangeCommonDim2048, LocalRangeCommonDim2048);
  
        e = queue.submit([&](handler& cgh) {
          cgh.parallel_for(
              RangeCommonDim2048, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                GEMV_Int4Weight_FP16InOutNx16Temp_largeGRF_block_ppg8_8T<2048, 3>(
                    (uint8_t*)w,
                    (uint8_t*)a,
                    (uint8_t*)dst,
                    (uint8_t*)s,
                    N,
                    ndi);
              });
        });
    }
    // else{
    //     std::cerr << "M = " << M << std::endl;
    //     std::cerr << "N = " << N << std::endl;
    //     std::cerr << "K = " << K << std::endl;
    // }
    // else{
    //     std::cerr << "OP NOT supportedin SYCL but directed to SYCL PATH!" << std::endl;
    // }
    //     else if (K==8192 && N==3072){
    //         uint32_t groupsV2 = (N + 8 - 1) / 8;
    //         ::sycl::range<1> GlobalRangeCommonDim8192(groupsV2 * 8);
    //         ::sycl::range<1> LocalRangeCommonDim8192(8);
    //         ::sycl::nd_range<1> RangeCommonDim8192(
    //             GlobalRangeCommonDim8192, LocalRangeCommonDim8192);
    
    //         e = queue.submit([&](handler& cgh) {
    //         cgh.parallel_for(
    //             RangeCommonDim8192, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
    //                 GEMV_Int4Weight_FP16InOutNx16Temp_largeGRF_block_ppg8_8T<8192, 3>(
    //                     (uint8_t*)w,
    //                     (uint8_t*)a,
    //                     (uint8_t*)dst,
    //                     (uint8_t*)s,
    //                     ndi);
    //             });
    //         });
    //     }
    //     else if (K == 3072 /* && N == 8192 || K == 3072 && N == 3072 || K == 3072 && N == 1024 */) {
    //         uint32_t groupsV2 = (N + 8 - 1) / 8;
    //         ::sycl::range<1> GlobalRangeCommonDim3072(groupsV2 * 6);
    //         ::sycl::range<1> LocalRangeCommonDim3072(6);
    //         ::sycl::nd_range<1> RangeCommonDim3072(
    //             GlobalRangeCommonDim3072, LocalRangeCommonDim3072);
    
    //         e = queue.submit([&](handler& cgh) {
    //         cgh.parallel_for(
    //             RangeCommonDim3072, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
    //                 GEMV_Int4Weight_FP16InOutNx16Temp_largeGRF_block_ppg8_8T<3072, 3>(
    //                     (uint8_t*)w,
    //                     (uint8_t*)a,
    //                     (uint8_t*)dst,
    //                     (uint8_t*)s,
    //                     ndi);
    //             });
    //         });
    //     }
    //     else if (K==2048){
    //         uint32_t groupsV2 = (N + 8 - 1) / 8;
    //         ::sycl::range<1> GlobalRangeCommonDim2048(groupsV2 * 4);
    //         ::sycl::range<1> LocalRangeCommonDim2048(4);
    //         ::sycl::nd_range<1> RangeCommonDim2048(
    //             GlobalRangeCommonDim2048, LocalRangeCommonDim2048);
    
    //         e = queue.submit([&](handler& cgh) {
    //         cgh.parallel_for(
    //             RangeCommonDim2048, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
    //                 // GEMV_Int4Weight_FP16InOutNx16Temp_largeGRF_block<3072, 4>(
    //                 GEMV_Int4Weight_FP16InOutNx16Temp_largeGRF_block_ppg8_8T<2048, 3>(
    //                     (uint8_t*)w,
    //                     (uint8_t*)a,
    //                     (uint8_t*)dst,
    //                     (uint8_t*)s,
    //                     ndi);
    //             });
    //         });
    //     }
    // }
    // else{ // GEMM
    //     e = queue.submit([=](::sycl::handler& cgh) {
    //         cgh.parallel_for(::sycl::range<2>(M, N), [=](::sycl::id<2> index) {
    //             const uint m = index[0];
    //             const uint n = index[1];
    //             using accum_t = typename ::sycl::half;
    //             accum_t accumulator = 0.0f;

    //             const uint dst_index = n + m*N;
    //             for (uint y = 0; y < K; ++y) {
    //                 const uint input0_offset = y + m*K;
    //                 const uint decomp_offset = (y / group_size) * N + n;
    //                 const uint filter_offset = y + n*K;

    //                 accum_t scale = s[decomp_offset];
    //                 const char packed = w[filter_offset / 2];
    //                 char v0;
    //                 char v1;
    //                 accum_t zp_val;

    //                 zp_val = static_cast<accum_t>(0.0f);
    //                 const char s_bit = packed & 0x08;
    //                 const char mask = s_bit > 0 ? 0xF0 : 0x00;
    //                 v0 = (packed & 0x0F) | mask;
    //                 v1 = packed >> 4;

    //                 accum_t unpacked = filter_offset % 2 == 0 ? v0 : v1;

    //                 accum_t filter_val = (unpacked - zp_val) * scale;
    //                 accumulator += a[input0_offset] * filter_val;
    //             }
    //             dst[dst_index] = accumulator;
    //         });
    //     });
    // }
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
        // auto& stream = dynamic_cast<ocl::ocl_stream&>(network.get_stream());
        // auto& engine = dynamic_cast<ocl::ocl_engine&>(network.get_engine());
        // ::sycl::context sycl_context = ::sycl::make_context<::sycl::backend::opencl>(engine.get_cl_context().get());
        // ::sycl::queue sycl_queue = ::sycl::make_queue<::sycl::backend::opencl>(stream.get_cl_queue().get(), sycl_context);

        auto& stream = downcast<ocl::sycl_stream>(network.get_stream());
        auto& engine = downcast<ocl::sycl_engine>(network.get_engine());
        ::sycl::context sycl_context = engine.get_sycl_context();
        ::sycl::queue& sycl_queue = stream.get_sycl_queue();
        auto end = std::chrono::high_resolution_clock::now();

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

        // void* in = static_cast<void*>(inputs[0]->buffer_ptr());
        // void* wei = static_cast<void*>(weights->buffer_ptr());
        // void* out = static_cast<void*>(output->buffer_ptr());
        // void* ds = static_cast<void*>(inputs[1]->buffer_ptr());
        // void* dzp = inputs.size() == 3 ? static_cast<void*>(inputs[2]->buffer_ptr()) : nullptr;

        const ::sycl::half* in = static_cast<const ::sycl::half*>(inputs[0]->buffer_ptr());
        const uint8_t* wei = static_cast<const uint8_t*>(weights->buffer_ptr());
        ::sycl::half* out = static_cast<::sycl::half*>(output->buffer_ptr());
        const ::sycl::half* ds = static_cast<const ::sycl::half*>(inputs[1]->buffer_ptr());

        if (print) {
            std::cerr << "in: " << params->input_layouts[0].to_short_string() << std::endl;
            std::cerr << "wei: " << params->weights_layout.value().to_short_string() << std::endl;
            std::cerr << "out: " << params->output_layouts[0].to_short_string() << std::endl;
            std::cerr << "scale: " << params->input_layouts[2].to_short_string() << std::endl;

            std::cerr << "M = " << M << std::endl;
            std::cerr << "N = " << N << std::endl;
            std::cerr << "K = " << K << std::endl;
            std::cerr << "groups_num = " << groups_num << std::endl;
            std::cerr << "group_size = " << group_size << std::endl;
            std::cerr << "in_t = " << in_t << std::endl;
            std::cerr << "wei_t = " << wei_t << std::endl;
            std::cerr << "out_t = " << out_t << std::endl;
            std::cerr << "ds_t = " << ds_t << std::endl;

            std::cerr << "in = " << in << std::endl;
            std::cerr << "wei = " << wei << std::endl;
            std::cerr << "out = " << out << std::endl;
            std::cerr << "ds = " << ds << std::endl;
        }

        // OPENVINO_ASSERT(inputs.size() >= 2);

        // auto dzp_scalar = desc->decompression_zero_point_scalar;

        // bool barrier = stream.get_queue_type() == QueueTypes::out_of_order;

        return to_ocl_event(stream, run_fc_q4_0_fp16out(sycl_queue, in, wei, ds, out, M, N, K, group_size, groups_num));
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params) {
        auto source_weights_layout = impl_params.get_input_layout(1);
        auto target_weights_layout = source_weights_layout;
        target_weights_layout.format = format::oiyx;
        // target_weights_layout.format = format::ioyx;

        return std::make_shared<WeightsReorderParams>(source_weights_layout, target_weights_layout);
    }

    static std::unique_ptr<primitive_impl> create(const fully_connected_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        return cldnn::make_unique<fully_connected_sycl>(engine, config, get_weights_reorder(impl_params));
    }
};


std::unique_ptr<primitive_impl> FCImplementationManagerSYCL::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<fully_connected>());
    return sycl::fully_connected_sycl::create(static_cast<const fully_connected_node&>(node), params);
}

}  // namespace sycl
}  // namespace cldnn
