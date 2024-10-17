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

#include "gemm_test.h"

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

// #define GEMM_FP16AW_CAL_REF

template<typename AType, typename WType, typename DType>
::sycl::event run_fc_fp16out(::sycl::queue& queue, const AType* a, const WType* w, DType* dst,
                              size_t M, size_t N, size_t K, uint8_t* shuffleTt, ::sycl::half* bias, ::sycl::half* residual) {            
    ::sycl::event e;
    // k=M, n=K, m=N

#if 1
    size_t alignedTokeSize = (M + 7) / 8;
    alignedTokeSize = alignedTokeSize * 8;
    int groupShuffleH = (K + 31) / 32;
    int groupShuffleV = alignedTokeSize / 8;
    int localShuffleH = 1;
    int localShuffleV = 1;
    ::sycl::range<2> GlobalRangeShuffle(groupShuffleH * localShuffleH, groupShuffleV * localShuffleV);
    ::sycl::range<2> LocalRangeShuffle(localShuffleH, localShuffleV);
    ::sycl::nd_range<2> RangeShuffle(GlobalRangeShuffle, LocalRangeShuffle);
    int groupReduce2048H = ((N + 255) / 256) * ((M + 255) / 256);
    int groupReduce2048V = 1;
    int localReduce2048H = 32;
    int localReduce2048V = 1;
    ::sycl::range<2> GlobalRangeReduce2048(groupReduce2048H * localReduce2048H, groupReduce2048V * localReduce2048V);
    ::sycl::range<2> LocalRangeReduce2048(localReduce2048H, localReduce2048V);
    ::sycl::nd_range<2> RangeReduce2048(GlobalRangeReduce2048, LocalRangeReduce2048);

    // std::cout << "M: " << M << std::endl;
    // std::cout << "K: " << K << std::endl;
    // std::cout << "N: " << N << std::endl;

#ifdef GEMM_FP16AW_CAL_REF
    e = queue.submit([&](handler& cgh) {
        cgh.parallel_for(RangeShuffle, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            fp16ShuffleToFp16_xmx_no_k_split_ref((uint8_t*)a, (uint8_t*)shuffleTt, K, M, ndi);
          });
        });
#else
    e = queue.submit([&](handler& cgh) {
        cgh.parallel_for(RangeShuffle, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            fp16ShuffleToFp16_xmx_no_k_split((uint8_t*)a, (uint8_t*)shuffleTt, K, M, ndi);
          });
        });
#endif

    if (K == 1280)
    {
#ifdef GEMM_FP16AW_CAL_REF
    e = queue.submit([&](handler& cgh) {
        cgh.parallel_for(RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            gemmReduce2048WeightsFP16InputShffuledFp16_xmx_ppifull_bb8_notmp_ctile<1280/32>((uint8_t*)w, (uint8_t*)shuffleTt, (uint8_t*)dst, N, K, M, ndi);
          });
        });
#else
    e = queue.submit([&](handler& cgh) {
        cgh.parallel_for(RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            gemmReduce2048WeightsFP16InputShffuledFp16_xmx_ppifull_bb8_notmp_int8cal_ctile<1280/32>((uint8_t*)w, (uint8_t*)shuffleTt, (uint8_t*)dst, N, K, M, ndi);
          });
        });
#endif
    }
    else if (K == 2048)
    {
#ifdef GEMM_FP16AW_CAL_REF
    e = queue.submit([&](handler& cgh) {
        cgh.parallel_for(RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            gemmReduce2048WeightsFP16InputShffuledFp16_xmx_ppifull_bb8_notmp_ctile<2048/32>((uint8_t*)w, (uint8_t*)shuffleTt, (uint8_t*)dst, N, K, M, ndi);
          });
        });
#else
    e = queue.submit([&](handler& cgh) {
        cgh.parallel_for(RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            gemmReduce2048WeightsFP16InputShffuledFp16_xmx_ppifull_bb8_notmp_int8cal_ctile<2048/32>((uint8_t*)w, (uint8_t*)shuffleTt, (uint8_t*)dst, N, K, M, ndi);
          });
        });
#endif
    }
    else if (K == 2560)
    {
#ifdef GEMM_FP16AW_CAL_REF
    e = queue.submit([&](handler& cgh) {
        cgh.parallel_for(RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            gemmReduce2048WeightsFP16InputShffuledFp16_xmx_ppifull_bb8_notmp_ctile<2560/32>((uint8_t*)w, (uint8_t*)shuffleTt, (uint8_t*)dst, N, K, M, ndi);
          });
        });
#else
    e = queue.submit([&](handler& cgh) {
        cgh.parallel_for(RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            gemmReduce2048WeightsFP16InputShffuledFp16_xmx_ppifull_bb8_notmp_int8cal_ctile<2560/32>((uint8_t*)w, (uint8_t*)shuffleTt, (uint8_t*)dst, N, K, M, ndi);
          });
        });
#endif
    }
    else if (K == 640)
    {
#ifdef GEMM_FP16AW_CAL_REF
    e = queue.submit([&](handler& cgh) {
        cgh.parallel_for(RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            gemmReduce2048WeightsFP16InputShffuledFp16_xmx_ppifull_bb8_notmp_ctile<640/32>((uint8_t*)w, (uint8_t*)shuffleTt, (uint8_t*)dst, N, K, M, ndi);
          });
        });
#else
    e = queue.submit([&](handler& cgh) {
        cgh.parallel_for(RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            gemmReduce2048WeightsFP16InputShffuledFp16_xmx_ppifull_bb8_notmp_int8cal_ctile<640/32>((uint8_t*)w, (uint8_t*)shuffleTt, (uint8_t*)dst, N, K, M, ndi);
          });
        });
#endif
    }
    else if (K == 5120)
    {
#ifdef GEMM_FP16AW_CAL_REF
    e = queue.submit([&](handler& cgh) {
        cgh.parallel_for(RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            gemmReduce2048WeightsFP16InputShffuledFp16_xmx_ppifull_bb8_notmp_ctile<5120/32>((uint8_t*)w, (uint8_t*)shuffleTt, (uint8_t*)dst, N, K, M, ndi);
          });
        });
#else
    e = queue.submit([&](handler& cgh) {
        cgh.parallel_for(RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            gemmReduce2048WeightsFP16InputShffuledFp16_xmx_ppifull_bb8_notmp_int8cal_ctile<5120/32>((uint8_t*)w, (uint8_t*)shuffleTt, (uint8_t*)dst, N, K, M, ndi);
          });
        });
#endif
    }


#else
     e = queue.submit([=](::sycl::handler& cgh) {
        cgh.parallel_for(::sycl::range<2>(M, N), [=](::sycl::id<2> index) {
            const uint m = index[0];
            const uint n = index[1];
            ::sycl::half accumulator = 0.0f;
            const uint dst_index = n + m*N;
            for (uint y = 0; y < K; ++y) {
                const uint input0_offset = y + m*K;
                const uint weight0_offset = y + n*K;
                accumulator += a[input0_offset] * w[weight0_offset];
            }
            dst[dst_index] = accumulator;
        });
     });
#endif

    if (bias != nullptr && residual != nullptr)
    {
        // post op fuse.
        // out (M,N)  bias (N)   residual  (M,N)
        queue.submit([&](::sycl::handler& cgh) {
            cgh.parallel_for(::sycl::range<2>(M, N), [=](::sycl::id<2> idx) {
                int i = idx[0];
                int j = idx[1];
                
                int32_t biasIdx = j;
                int32_t index = i * N + j;
                
                dst[index] = dst[index] + bias[biasIdx] + residual[index];
            });
        });
    }
    else if (residual != nullptr)
    {
        // post op fuse.
        // out (M,N)  bias (N)   residual  (M,N)
        queue.submit([&](::sycl::handler& cgh) {
            cgh.parallel_for(::sycl::range<2>(M, N), [=](::sycl::id<2> idx) {
                int i = idx[0];
                int j = idx[1];
                
                int32_t biasIdx = j;
                int32_t index = i * N + j;
                
                dst[index] = dst[index] + residual[index];
            });
        });
    }
    else if (bias != nullptr)
    {
        // post op fuse.
        // out (M,N)  bias (N)   residual  (M,N)
        queue.submit([&](::sycl::handler& cgh) {
            cgh.parallel_for(::sycl::range<2>(M, N), [=](::sycl::id<2> idx) {
                int i = idx[0];
                int j = idx[1];
                
                int32_t biasIdx = j;
                int32_t index = i * N + j;
                
                dst[index] = dst[index] + bias[biasIdx];
            });
        });
    }

    free(shuffleTt, queue);
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
        std::vector<memory::ptr> inputs = { instance.input_memory_ptr(0) };
        size_t in_id = instance.bias_term() ? 3 : 2;

        std::vector<memory::ptr> fused_inputs;
        if (instance.has_fused_primitives()){
            std::cout << "input mem count:" << instance.inputs_memory_count() << std::endl;
            std::cout << "fused mem count:" << instance.get_fused_mem_count() << std::endl;
            fused_inputs = {instance.fused_memory(0)};
        }
        // OPENVINO_ASSERT(!instance.bias_term() && !instance.get_node().has_fused_primitives());
        ov::element::Type_t in_t = params->input_layouts[0].data_type;
        ov::element::Type_t wei_t = params->weights_layout.value().data_type;
        ov::element::Type_t out_t = params->output_layouts[0].data_type;
        
        size_t M = out_shape[1];
        size_t N = out_shape[2];
        size_t K = params->weights_layout.value().get_partial_shape()[1].get_length();

        auto sycl_device = sycl_queue.get_info<info::queue::device>();
        uint8_t* shuffleTt = static_cast<uint8_t*>(aligned_alloc_device(4096, (M+255)/256*256 * K * sizeof(::sycl::half), sycl_device, sycl_context));
        // unsigned char* hostshuffleTt = (unsigned char *)malloc((M+8) * K * sizeof(::sycl::half));
        // memset(hostshuffleTt, 0, (M+8) * K * sizeof(::sycl::half));
        // sycl_queue.memcpy(shuffleTt, hostshuffleTt, (M+8) * K * sizeof(::sycl::half)).wait();
        // free(hostshuffleTt);
        // uint8_t* shuffleTt = nullptr;

        // void* in = static_cast<void*>(inputs[0]->buffer_ptr());
        // void* wei = static_cast<void*>(weights->buffer_ptr());
        // void* out = static_cast<void*>(output->buffer_ptr());
        if (print) {
            std::cerr << "in: " << params->input_layouts[0].to_short_string() << std::endl;
            std::cerr << "wei: " << params->weights_layout.value().to_short_string() << std::endl;
            std::cerr << "out: " << params->output_layouts[0].to_short_string() << std::endl;
            
            std::cerr << "M = " << M << std::endl;
            std::cerr << "N = " << N << std::endl;
            std::cerr << "K = " << K << std::endl;
            
            std::cerr << "in_t = " << in_t << std::endl;
            std::cerr << "wei_t = " << wei_t << std::endl;
            std::cerr << "out_t = " << out_t << std::endl;

            // std::cerr << "in = " << in << std::endl;
            // std::cerr << "wei = " << wei << std::endl;
            // std::cerr << "out = " << out << std::endl;
        }
        // OPENVINO_ASSERT(inputs.size() >= 2);
        bool barrier = stream.get_queue_type() == QueueTypes::out_of_order;

        const ::sycl::half* in = static_cast<const ::sycl::half*>(inputs[0]->buffer_ptr());
        const ::sycl::half* wei = static_cast<const ::sycl::half*>(weights->buffer_ptr());
        ::sycl::half* residual = nullptr;
        ::sycl::half* b = nullptr;
        if (instance.has_fused_primitives()){
            residual = static_cast<::sycl::half*>(fused_inputs[0]->buffer_ptr());
            b = static_cast<::sycl::half*>(bias->buffer_ptr());
        }
        ::sycl::half* out = static_cast<::sycl::half*>(output->buffer_ptr());
        return to_ocl_event(stream, run_fc_fp16out(sycl_queue, in, wei, out, M, N, K, shuffleTt, b, residual));
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