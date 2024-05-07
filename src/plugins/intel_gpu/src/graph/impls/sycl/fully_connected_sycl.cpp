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

#include <algorithm>
#include <chrono>
#include <memory>

#include <ext/intel/esimd.hpp>
using fp16 = ::sycl::half;

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
/*
template<typename AType, typename WType, typename ZPType, typename ScaleType, typename DType>
::sycl::event run_fc_int4_woq(::sycl::queue& queue, bool enqueue_barrier, const AType* a, const WType* w, const ZPType* zp, const ScaleType* s, DType* dst,
                              size_t M, size_t N, size_t K, size_t group_size, size_t groups_num, const ov::Shape& out_shape, optional_value<float> dzp_s) {
    if (enqueue_barrier) {
        queue.submit([=](::sycl::handler& cgh) {
            cgh.ext_oneapi_barrier();
        });
    }

    bool has_value = dzp_s.has_value();
    float dzp_value = dzp_s.value_or(0.0f);
    return queue.submit([=](::sycl::handler& cgh) {
        cgh.parallel_for(::sycl::range<3>(out_shape[0], out_shape[1], out_shape[2]), [=](::sycl::id<3> index) {
            const uint b = index[0];
            const uint m = index[1];
            const uint n = index[2];
            using accum_t = typename AccumulatorType<AType, WType>::type;
            accum_t accumulator = 0.0f;

            const uint dst_index = n + m*N + b*N*M;
            for (uint y = 0; y < K; ++y) {
                const uint input0_offset = y + m*K + b*M*K;
                const uint decomp_offset = (y / group_size % groups_num)*N + n % N;
                const uint filter_offset = y + n*K;
                // const uint filter_offset = y*N + n;
                const uint zp_offset = 0;


                accum_t zp_val = has_value ? static_cast<accum_t>(dzp_value) : static_cast<accum_t>(zp[zp_offset]);
                accum_t scale = s[decomp_offset];
                const WType packed = w[filter_offset / 2];

                const WType v0 = packed & 0x0F;
                const WType v1 = (packed & 0xF0) >> 4;
                accum_t unpacked = filter_offset % 2 == 0 ? v0 : v1;

                accum_t filter_val = (unpacked - zp_val) * scale;
                accumulator += a[input0_offset] * filter_val;
            }
            dst[dst_index] = accumulator;
        });
    });
}

template<typename AType, typename WType, typename ZPType, typename ScaleType, typename DType>
::sycl::event run_fc_int8_woq(::sycl::queue& queue, bool enqueue_barrier, const AType* a, const WType* w, const ZPType* zp, const ScaleType* s, DType* dst,
                     size_t M, size_t N, size_t K, size_t group_size, size_t groups_num, const ov::Shape& out_shape, optional_value<float> dzp_s) {
    if (enqueue_barrier) {
        queue.submit([=](::sycl::handler& cgh) {
            cgh.ext_oneapi_barrier();
        });
    }

    bool has_value = dzp_s.has_value();
    float dzp_value = dzp_s.value_or(0.0f);

    return queue.submit([=](::sycl::handler& cgh) {
        cgh.parallel_for(::sycl::range<3>(out_shape[0], out_shape[1], out_shape[2]), [=](::sycl::id<3> index) {
            const uint b = index[0];
            const uint m = index[1];
            const uint n = index[2];
            using accum_t = typename AccumulatorType<AType, WType>::type;
            accum_t accumulator = 0.0f;

            for (uint y = 0; y < K; ++y) {
                const uint input0_offset = y + m*K + b*M*K;
                const uint zp_offset = (y / group_size % groups_num)*N + n % N;
                const uint decomp_offset = (y / group_size % groups_num)*N + n % N;
                const uint filter_offset = y + n*K;
                // const uint filter_offset = y*N + n;

                accum_t zp_val = has_value ? static_cast<accum_t>(dzp_value) : static_cast<accum_t>(zp[zp_offset]);
                accum_t scale = s[decomp_offset];
                accum_t filter_compressed = static_cast<accum_t>(w[filter_offset]);
                accum_t filter_val = (filter_compressed - zp_val) * scale;
                accumulator += a[input0_offset] * filter_val;
            }
            const uint dst_index = n + m*N + b*N*M;
            dst[dst_index] = accumulator;
        });
    });
}
*/

using namespace ::sycl::ext::intel::esimd;
using namespace ::sycl;
using namespace ::sycl::ext::intel::esimd;
using namespace ::sycl::ext::intel::esimd::xmx;
ESIMD_INLINE void gemmCommonDim4096Fp16V0(uint8_t* a, uint8_t* b, uint8_t* c, int tokenSize, sycl::nd_item<2>& ndi) {
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    constexpr uint32_t baseOffsetInc4[4] = { 0, 1, 2, 3 };
    __ESIMD_NS::slm_init(64 * 4 * 16 * 2 * sizeof(fp16));
    int hh = ndi.get_local_linear_id(); // [0, 64)
    int h = ndi.get_group(0); // [0, (row + 15) / 16)
    int v = ndi.get_group(1); // [0, (row + 15) / 16)
    uint32_t tokenOffset = v * 128;
    int outputRow = ndi.get_group_range(0) * 16;
    uint32_t offsetA = (h * 64 + hh) * 64 * 16 * sizeof(fp16);
    uint32_t offsetB = hh * 64 * sizeof(fp16) + 4096 * tokenOffset * sizeof(fp16);
    uint32_t offsetC = h * 16 + tokenOffset * outputRow + hh * outputRow;
    simd<fp16, 8 * 64> bb(0.0f);
    simd<fp16, 64 * 16> aa;
    simd<fp16, 4 * 16> cc(0.0f);
    simd<uint32_t, 4> columnIdx(baseOffsetInc4);
    columnIdx += tokenOffset;
    uint32_t loopCount = 32;
    if (v == ndi.get_group_range(1) - 1) {
        loopCount = tokenSize - tokenOffset;
        loopCount = (loopCount + 3) >> 2;
    }

#pragma unroll
    for (int k = 0; k < 8; k++) {
        aa.template bit_cast_view<uint8_t>().template select<256, 1>(256 * k) =
            __ESIMD_ENS::lsc_block_load<
            unsigned char,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::none,
            __ESIMD_ENS::cache_hint::none>((uint8_t*)a + offsetA + 256 * k);
    }

#pragma unroll
    for (int k = 0; k < 4; k++) {
        bb.template bit_cast_view<unsigned char>().template select<128, 1>(128 * k) =
            __ESIMD_ENS::lsc_block_load<
            uint8_t,
            128,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::none,
            __ESIMD_ENS::cache_hint::none>((uint8_t*)b + offsetB);
        offsetB += 4096 * sizeof(fp16);
    }

    for (int nn = 0; nn < loopCount; nn++) {
        cc = 0;
#pragma unroll
        for (int k = 0; k < 64; k++) {
            cc.select<16, 1>(16 * 0) += aa.select<16, 1>(16 * k) * bb[64 * 0 + k];
        }

#pragma unroll
        for (int k = 0; k < 64; k++) {
            cc.select<16, 1>(16 * 1) += aa.select<16, 1>(16 * k) * bb[64 * 1 + k];
        }


#pragma unroll
        for (int k = 0; k < 64; k++) {
            cc.select<16, 1>(16 * 2) += aa.select<16, 1>(16 * k) * bb[64 * 2 + k];
        }


#pragma unroll
        for (int k = 0; k < 64; k++) {
            cc.select<16, 1>(16 * 3) += aa.select<16, 1>(16 * k) * bb[64 * 3 + k];
        }
        slm_block_store<fp16, 16>((hh * 16 + 0 * 16 * 64) * sizeof(fp16), cc.select<16, 1>(16 * 0));
        slm_block_store<fp16, 16>((hh * 16 + 1 * 16 * 64) * sizeof(fp16), cc.select<16, 1>(16 * 1));
        slm_block_store<fp16, 16>((hh * 16 + 2 * 16 * 64) * sizeof(fp16), cc.select<16, 1>(16 * 2));
        slm_block_store<fp16, 16>((hh * 16 + 3 * 16 * 64) * sizeof(fp16), cc.select<16, 1>(16 * 3));

        barrier();

        if (hh < 4) {
            if (columnIdx[hh] < tokenSize) {
                uint32_t slmOffset = hh * 16 * 64 * sizeof(fp16);
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    bb.template bit_cast_view<fp16>().template select<128, 1>(128 * k) = slm_block_load<fp16, 128>(slmOffset + k * 128 * sizeof(fp16));
                }
#pragma unroll
                for (int k = 1; k < 16; k++) {
                    bb.template bit_cast_view<fp16>().template select<32, 1>(0) += bb.template bit_cast_view<fp16>().template select<32, 1>(32 * k);
                }
                cc.select<16, 1>(0) = bb.template bit_cast_view<fp16>().template select<16, 1>(0) + bb.template bit_cast_view<fp16>().template select<16, 1>(16);

#pragma unroll
                for (int k = 0; k < 4; k++) {
                    bb.template bit_cast_view<fp16>().template select<128, 1>(128 * k) = slm_block_load<fp16, 128>(slmOffset + 4 * 128 * sizeof(fp16) + k * 128 * sizeof(fp16));
                }

#pragma unroll
                for (int k = 1; k < 16; k++) {
                    bb.template bit_cast_view<fp16>().template select<32, 1>(0) += bb.template bit_cast_view<fp16>().template select<32, 1>(32 * k);
                }
                bb.template bit_cast_view<fp16>().template select<16, 1>(0) += bb.template bit_cast_view<fp16>().template select<16, 1>(16);
                cc.select<16, 1>(0) += bb.template bit_cast_view<fp16>().template select<16, 1>(0);
                __ESIMD_ENS::lsc_block_store<
                    fp16,
                    16,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::write_back,
                    __ESIMD_ENS::cache_hint::write_back>((fp16*)c + offsetC, cc.select<16, 1>(0));
                offsetC += 4 * outputRow;
            }
        }

        columnIdx += 4;
        //barrier();

        if (nn != loopCount - 1) {
#pragma unroll
            for (int k = 0; k < 4; k++) {
                bb.template bit_cast_view<unsigned char>().template select<128, 1>(128 * k) =
                    __ESIMD_ENS::lsc_block_load<
                    uint8_t,
                    128,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
                offsetB += 4096 * sizeof(fp16);
            }

        }
        //barrier();
    }
}

ESIMD_INLINE void gemmCommonDim4096Fp16V1(uint8_t* a, uint8_t* b, uint8_t* c, int tokenSize, sycl::nd_item<2>& ndi) {
    constexpr uint8_t baseOffsetInc16[16] = { 0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15 };
    constexpr uint8_t baseOffsetInc4[4] = { 0, 1, 2, 3 };
    __ESIMD_NS::slm_init(64 * 4 * 16 * 2 * sizeof(fp16));
    int hh = ndi.get_local_linear_id(); // [0, 64)
    int h = ndi.get_group(0); // [0, (row + 15) / 16)
    int v = ndi.get_group(1); // [0, (row + 15) / 16)
    uint32_t tokenOffset = 0;
    int outputRow = ndi.get_group_range(0) * 16;
    uint32_t offsetA = (h * 64 + hh) * 64 * 16 * sizeof(fp16);
    uint32_t offsetB = hh * 64 * sizeof(fp16) + 4096 * tokenOffset * sizeof(fp16);
    uint32_t offsetC = h * 16 + tokenOffset * outputRow + hh * outputRow;
    simd<fp16, 8 * 64> bb(0.0f);
    simd<fp16, 64 * 16> aa;
    simd<fp16, 4 * 16> cc(0.0f);
    simd<uint8_t, 4> columnIdx(baseOffsetInc4);
    columnIdx += tokenOffset;
#if 0
    uint32_t loopCount = 8;
    if (v == ndi.get_group_range(1) - 1) {
        loopCount = tokenSize - tokenOffset;
        loopCount = (loopCount + 15) >> 4;
    }
#else
    uint32_t loopCount = (tokenSize + 3) >> 2;

#endif
#pragma unroll
    for (int k = 0; k < 8; k++) {
        aa.template bit_cast_view<uint8_t>().template select<256, 1>(256 * k) =
            __ESIMD_ENS::lsc_block_load<
            unsigned char,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA + 256 * k);
    }


    for (int nn = 0; nn < loopCount; nn++) {

        cc = 0;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            bb.template bit_cast_view<unsigned char>().template select<128, 1>(128 * k) =
                __ESIMD_ENS::lsc_block_load<
                uint8_t,
                128,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
            offsetB += 4096 * sizeof(fp16);
        }

#pragma unroll
        for (int k = 0; k < 64; k++) {
            cc.select<16, 1>(16 * 0) += aa.select<16, 1>(16 * k) * bb[64 * 0 + k];
            cc.select<16, 1>(16 * 1) += aa.select<16, 1>(16 * k) * bb[64 * 1 + k];
            cc.select<16, 1>(16 * 2) += aa.select<16, 1>(16 * k) * bb[64 * 2 + k];
            cc.select<16, 1>(16 * 3) += aa.select<16, 1>(16 * k) * bb[64 * 3 + k];
        }


        int slm_offset = hh * 16 + 64 * 16 * 4 * (nn & 1);

        slm_block_store<fp16, 16>((slm_offset + 0 * 16 * 64) * sizeof(fp16), cc.select<16, 1>(16 * 0));
        slm_block_store<fp16, 16>((slm_offset + 1 * 16 * 64) * sizeof(fp16), cc.select<16, 1>(16 * 1));
        slm_block_store<fp16, 16>((slm_offset + 2 * 16 * 64) * sizeof(fp16), cc.select<16, 1>(16 * 2));
        slm_block_store<fp16, 16>((slm_offset + 3 * 16 * 64) * sizeof(fp16), cc.select<16, 1>(16 * 3));


        barrier();

        if (hh < 4) {
            if (columnIdx[hh] < tokenSize) {
                uint32_t slmOffset = hh * 16 * 64 * sizeof(fp16) + 64 * 4 * 16 * sizeof(fp16) * (nn & 1);
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    bb.template bit_cast_view<fp16>().template select<128, 1>(128 * k) = slm_block_load<fp16, 128>(slmOffset + k * 128 * sizeof(fp16));
                }
#pragma unroll
                for (int k = 1; k < 16; k++) {
                    bb.template bit_cast_view<fp16>().template select<32, 1>(0) += bb.template bit_cast_view<fp16>().template select<32, 1>(32 * k);
                }
                cc.select<16, 1>(0) = bb.template bit_cast_view<fp16>().template select<16, 1>(0) + bb.template bit_cast_view<fp16>().template select<16, 1>(16);

#pragma unroll
                for (int k = 0; k < 4; k++) {
                    bb.template bit_cast_view<fp16>().template select<128, 1>(128 * k) = slm_block_load<fp16, 128>(slmOffset + 4 * 128 * sizeof(fp16) + k * 128 * sizeof(fp16));
                }

#pragma unroll
                for (int k = 1; k < 16; k++) {
                    bb.template bit_cast_view<fp16>().template select<32, 1>(0) += bb.template bit_cast_view<fp16>().template select<32, 1>(32 * k);
                }
                bb.template bit_cast_view<fp16>().template select<16, 1>(0) += bb.template bit_cast_view<fp16>().template select<16, 1>(16);
                cc.select<16, 1>(0) += bb.template bit_cast_view<fp16>().template select<16, 1>(0);
                __ESIMD_ENS::lsc_block_store<
                    fp16,
                    16,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::write_back,
                    __ESIMD_ENS::cache_hint::write_back>((fp16*)c + offsetC, cc.select<16, 1>(0));
                offsetC += 4 * outputRow;
            }
        }

        columnIdx += 4;
        //   barrier();
    }
}

ESIMD_INLINE void gemmCommonDim4096Fp16V1_fp32out(uint8_t* a, uint8_t* b, uint8_t* c, int tokenSize, sycl::nd_item<2>& ndi) {
    constexpr uint8_t baseOffsetInc16[16] = { 0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15 };
    constexpr uint8_t baseOffsetInc4[4] = { 0, 1, 2, 3 };
    __ESIMD_NS::slm_init(64 * 4 * 16 * 2 * sizeof(fp16));
    int hh = ndi.get_local_linear_id(); // [0, 64)
    int h = ndi.get_group(0); // [0, (row + 15) / 16)
    int v = ndi.get_group(1); // [0, (row + 15) / 16)
    uint32_t tokenOffset = 0;
    int outputRow = ndi.get_group_range(0) * 16;
    uint32_t offsetA = (h * 64 + hh) * 64 * 16 * sizeof(fp16);
    uint32_t offsetB = hh * 64 * sizeof(fp16) + 4096 * tokenOffset * sizeof(fp16);
    uint32_t offsetC = h * 16 + tokenOffset * outputRow + hh * outputRow;
    simd<fp16, 8 * 64> bb(0.0f);
    simd<fp16, 64 * 16> aa;
    simd<fp16, 4 * 16> cc(0.0f);
    simd<uint8_t, 4> columnIdx(baseOffsetInc4);
    columnIdx += tokenOffset;
#if 0
    uint32_t loopCount = 8;
    if (v == ndi.get_group_range(1) - 1) {
        loopCount = tokenSize - tokenOffset;
        loopCount = (loopCount + 15) >> 4;
    }
#else
    uint32_t loopCount = (tokenSize + 3) >> 2;

#endif
#pragma unroll
    for (int k = 0; k < 8; k++) {
        aa.template bit_cast_view<uint8_t>().template select<256, 1>(256 * k) =
            __ESIMD_ENS::lsc_block_load<
            unsigned char,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA + 256 * k);
    }


    for (int nn = 0; nn < loopCount; nn++) {

        cc = 0;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            bb.template bit_cast_view<unsigned char>().template select<128, 1>(128 * k) =
                __ESIMD_ENS::lsc_block_load<
                uint8_t,
                128,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);
            offsetB += 4096 * sizeof(fp16);
        }

#pragma unroll
        for (int k = 0; k < 64; k++) {
            cc.select<16, 1>(16 * 0) += aa.select<16, 1>(16 * k) * bb[64 * 0 + k];
            cc.select<16, 1>(16 * 1) += aa.select<16, 1>(16 * k) * bb[64 * 1 + k];
            cc.select<16, 1>(16 * 2) += aa.select<16, 1>(16 * k) * bb[64 * 2 + k];
            cc.select<16, 1>(16 * 3) += aa.select<16, 1>(16 * k) * bb[64 * 3 + k];
        }


        int slm_offset = hh * 16 + 64 * 16 * 4 * (nn & 1);

        slm_block_store<fp16, 16>((slm_offset + 0 * 16 * 64) * sizeof(fp16), cc.select<16, 1>(16 * 0));
        slm_block_store<fp16, 16>((slm_offset + 1 * 16 * 64) * sizeof(fp16), cc.select<16, 1>(16 * 1));
        slm_block_store<fp16, 16>((slm_offset + 2 * 16 * 64) * sizeof(fp16), cc.select<16, 1>(16 * 2));
        slm_block_store<fp16, 16>((slm_offset + 3 * 16 * 64) * sizeof(fp16), cc.select<16, 1>(16 * 3));


        barrier();

        if (hh < 4) {
            if (columnIdx[hh] < tokenSize) {
                uint32_t slmOffset = hh * 16 * 64 * sizeof(fp16) + 64 * 4 * 16 * sizeof(fp16) * (nn & 1);
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    bb.template bit_cast_view<fp16>().template select<128, 1>(128 * k) = slm_block_load<fp16, 128>(slmOffset + k * 128 * sizeof(fp16));
                }
#pragma unroll
                for (int k = 1; k < 16; k++) {
                    bb.template bit_cast_view<fp16>().template select<32, 1>(0) += bb.template bit_cast_view<fp16>().template select<32, 1>(32 * k);
                }
                cc.select<16, 1>(0) = bb.template bit_cast_view<fp16>().template select<16, 1>(0) + bb.template bit_cast_view<fp16>().template select<16, 1>(16);

#pragma unroll
                for (int k = 0; k < 4; k++) {
                    bb.template bit_cast_view<fp16>().template select<128, 1>(128 * k) = slm_block_load<fp16, 128>(slmOffset + 4 * 128 * sizeof(fp16) + k * 128 * sizeof(fp16));
                }

#pragma unroll
                for (int k = 1; k < 16; k++) {
                    bb.template bit_cast_view<fp16>().template select<32, 1>(0) += bb.template bit_cast_view<fp16>().template select<32, 1>(32 * k);
                }
                bb.template bit_cast_view<fp16>().template select<16, 1>(0) += bb.template bit_cast_view<fp16>().template select<16, 1>(16);
                cc.select<16, 1>(0) += bb.template bit_cast_view<fp16>().template select<16, 1>(0);
                __ESIMD_ENS::lsc_block_store<
                    float,
                    16,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::write_back,
                    __ESIMD_ENS::cache_hint::write_back>((float*)c + offsetC, cc.select<16, 1>(0));
                offsetC += 4 * outputRow;
            }
        }

        columnIdx += 4;
        //   barrier();
    }
}

ESIMD_INLINE void gemmCommonDim11008Fp16NoReshape(uint8_t* a, uint8_t* b, uint8_t* c, int tokenSize, nd_item<2>& ndi) {
  constexpr uint32_t baseOffsetInc4[4] = { 0, 1, 2, 3 };
  constexpr uint32_t baseOffsetInc2[4] = { 0, 1};
  __ESIMD_NS::slm_init(4 * 12 * 4 * sizeof(fp16));
  int hh = ndi.get_local_id(0); // [0, 11)
  int vv = ndi.get_local_id(1); // [0, 4)
  int h = ndi.get_group(0); // [0, (row + 15) / 16)
  int outputRow = ndi.get_group_range(0) * 4;
  int localLinearId = vv * 11 + hh;
  uint32_t offsetA = h * 11008 * 4 * sizeof(fp16) + vv * 11008 * sizeof(fp16) + hh * 128 * sizeof(fp16);
  uint32_t offsetB = hh * 128 * sizeof(fp16);
  uint32_t offsetC = h * 4;
  simd<fp16, 4 * 128> bb(0.0f);
  simd<fp16, 1024> aa;
  simd<fp16, 4 * 16> cc(0.0f);
  simd<uint32_t, 4> columnIdx(baseOffsetInc4);
  simd<uint32_t, 2> slmScatterOffsets(baseOffsetInc2);
  simd<uint32_t, 2> slmScatterOffsetsZeroPadding(baseOffsetInc2);
  slmScatterOffsetsZeroPadding = 88 + slmScatterOffsetsZeroPadding * 4;
  slmScatterOffsetsZeroPadding = slmScatterOffsetsZeroPadding + vv * 96;
  slmScatterOffsets = slmScatterOffsets + hh * 2;
  slmScatterOffsets = slmScatterOffsets * 4 + vv * 96;
  uint32_t loopCount = (tokenSize + 3) >> 2;

#pragma unroll
  for (int k = 0; k < 7; k++) {
    aa.template bit_cast_view<uint8_t>().template select<256, 1>(256 * k) =
      __ESIMD_ENS::lsc_block_load<
      unsigned char,
      256,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::uncached>((uint8_t*)a + offsetA);
    offsetA += 128 * 11 * sizeof(fp16);
  }

  if (hh < 9) {
#pragma unroll
    for (int k = 0; k < 1; k++) {
      aa.template bit_cast_view<uint8_t>().template select<256, 1>(256 * k + 256 * 7) =
        __ESIMD_ENS::lsc_block_load<
        unsigned char,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::uncached>((uint8_t*)a + offsetA);
      offsetA += 128 * 11 * sizeof(fp16);
    }
  } else {
    for (int k = 0; k < 1; k++) {
      aa.template bit_cast_view<uint8_t>().template select<256, 1>(256 * k + 256 * 7) = 0;
      offsetA += 128 * 11 * sizeof(fp16);
    }
  }

  for (int nn = 0; nn < loopCount; nn++) {
    cc = 0;
    offsetB = hh * 128 * sizeof(fp16) + nn * 4 * 11008 * sizeof(fp16);
#pragma unroll
    for (int nnn = 0; nnn < 7; nnn++) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        if (columnIdx[k] < tokenSize) {
          bb.template bit_cast_view<unsigned char>().template select<256, 1>(256 * k) =
            __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 11008 * k * sizeof(fp16));
        }
      }

      if (columnIdx[0] < tokenSize) {
#pragma unroll
        for (int k = 0; k < 8; k++) {
          cc.select<16, 1>(16 * 0) += aa.select<16, 1>(16 * k + 128 * nnn) * bb.select<16, 1>(16 * k + 128 * 0);
        }
      }

      if (columnIdx[1] < tokenSize) {
#pragma unroll
        for (int k = 0; k < 8; k++) {
          cc.select<16, 1>(16 * 1) += aa.select<16, 1>(16 * k + 128 * nnn) * bb.select<16, 1>(16 * k + 128 * 1);
        }
      }

      if (columnIdx[2] < tokenSize) {
#pragma unroll
        for (int k = 0; k < 8; k++) {
          cc.select<16, 1>(16 * 2) += aa.select<16, 1>(16 * k + 128 * nnn) * bb.select<16, 1>(16 * k + 128 * 2);
        }
      }

      if (columnIdx[3] < tokenSize) {
#pragma unroll
        for (int k = 0; k < 8; k++) {
          cc.select<16, 1>(16 * 3) += aa.select<16, 1>(16 * k + 128 * nnn) * bb.select<16, 1>(16 * k + 128 * 3);
        }
      }

      offsetB += 128 * 11 * sizeof(half);
    }

    if (hh < 9) {
#pragma unroll
      for (int nnn = 0; nnn < 1; nnn++) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
          if (columnIdx[k] < tokenSize) {
            bb.template bit_cast_view<unsigned char>().template select<256, 1>(256 * k) =
              __ESIMD_ENS::lsc_block_load<
              uint8_t,
              256,
              __ESIMD_ENS::lsc_data_size::default_size,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB + 11008 * k * sizeof(fp16));
          }
        }

        if (columnIdx[0] < tokenSize) {
#pragma unroll
          for (int k = 0; k < 8; k++) {
            cc.select<16, 1>(16 * 0) += aa.select<16, 1>(16 * k + 128 * nnn + 128 * 7) * bb.select<16, 1>(16 * k + 128 * 0);
          }
        }

        if (columnIdx[1] < tokenSize) {
#pragma unroll
          for (int k = 0; k < 8; k++) {
            cc.select<16, 1>(16 * 1) += aa.select<16, 1>(16 * k + 128 * nnn + 128 * 7) * bb.select<16, 1>(16 * k + 128 * 1);
          }
        }

        if (columnIdx[2] < tokenSize) {
#pragma unroll
          for (int k = 0; k < 8; k++) {
            cc.select<16, 1>(16 * 2) += aa.select<16, 1>(16 * k + 128 * nnn + 128 * 7) * bb.select<16, 1>(16 * k + 128 * 2);
          }
        }

        if (columnIdx[3] < tokenSize) {
#pragma unroll
          for (int k = 0; k < 8; k++) {
            cc.select<16, 1>(16 * 3) += aa.select<16, 1>(16 * k + 128 * nnn + 128 * 7) * bb.select<16, 1>(16 * k + 128 * 3);
          }
        }
        offsetB += 128 * 11 * sizeof(half);
      }
    }

#pragma unroll
    for (int k = 0; k < 4; k++) {
      cc.select<8, 1>(16 * k) += cc.select<8, 1>(16 * k + 8);
      cc.select<4, 1>(16 * k) += cc.select<4, 1>(16 * k + 4);
      cc.select<2, 1>(16 * k) += cc.select<2, 1>(16 * k + 2);
    }

    cc[0] = cc[0 * 16] + cc[0 * 16 + 1];
    cc[1] = cc[1 * 16] + cc[1 * 16 + 1];
    cc[2] = cc[2 * 16] + cc[2 * 16 + 1];
    cc[3] = cc[3 * 16] + cc[3 * 16 + 1];
    float temp0;
    float temp1;
    temp0 = cc.template bit_cast_view<float>()[0];
    temp1 = cc.template bit_cast_view<float>()[1];
    slm_scalar_store(slmScatterOffsets[0], temp0);
    slm_scalar_store(slmScatterOffsets[1], temp1);
    if (hh == 0) {
      float temp = 0;
      slm_scalar_store(slmScatterOffsetsZeroPadding[0], temp);
      slm_scalar_store(slmScatterOffsetsZeroPadding[1], temp);
    }

    barrier();
    if (localLinearId == 0) {
      bb.select<128, 1>(0) = slm_block_load<fp16, 128>(0);
      bb.select<64, 1>(128) = slm_block_load<fp16, 64>(128 * sizeof(fp16));
#pragma unroll
      for (int k = 0; k < 4; k++) {
        bb.select<16, 1>(48 * k) += bb.select<16, 1>(48 * k + 16);
        bb.select<16, 1>(48 * k) += bb.select<16, 1>(48 * k + 32);
        bb.select<8, 1>(48 * k) += bb.select<8, 1>(48 * k + 8);
        bb.select<4, 1>(192 + 4 * k) = bb.select<4, 1>(48 * k) + bb.select<4, 1>(48 * k + 4);
      }

#pragma unroll
      for (int k = 0; k < 4; k++) {
        if (columnIdx[k] < tokenSize) {
          bb.select<4, 1>(16 * k) = bb.select<4, 4>(192 + k);
          __ESIMD_ENS::lsc_block_store<
            float,
            4,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>((float*)c + offsetC + outputRow * k, bb.select<4, 1>(16 * k));
        }
      }
    }

    offsetC += outputRow * 4;
    columnIdx += 4;
  }
}



template<typename AType, typename WType, typename DType>
::sycl::event run_fc_fp16(::sycl::queue& queue, const AType* a, const WType* w, DType* dst,
                     size_t batch_size, size_t output_size, size_t K, const ov::element::Type_t& out_t) {
    
    // auto start = std::chrono::high_resolution_clock::now();
    // auto eb=queue.submit([=](::sycl::handler& cgh) {
    //     cgh.ext_oneapi_barrier();
    // });
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "barrier launch cost time: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " us" << std::endl;

    // K=4096
    int groupH = (output_size + 15) / 16;
    int groupV = 1;
    int localH = 64;
    int localV = 1;
    if (K==11008){
        groupH = (output_size + 3) / 4;
        groupV = 1;
        localH = 11;
        localV = 4;
    }

    ::sycl::range<2> GlobalRange(groupH * localH, groupV * localV);
    ::sycl::range<2> LocalRange(localH, localV);
    ::sycl::nd_range<2> Range(GlobalRange, LocalRange);
    // start = std::chrono::high_resolution_clock::now();
    ::sycl::event e;
    if (K==4096 && out_t == ov::element::f16){
        e = queue.submit([&](::sycl::handler& handle) {
                            handle.parallel_for(Range,
                            [=](::sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                                gemmCommonDim4096Fp16V1((uint8_t*)w, (uint8_t*)a, (uint8_t*)dst, batch_size, ndi);
                            });
                            });
    }
    else if (K==4096 && out_t == ov::element::f32){
        e = queue.submit([&](::sycl::handler& handle) {
                            handle.parallel_for(Range,
                            [=](::sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                                gemmCommonDim4096Fp16V1_fp32out((uint8_t*)w, (uint8_t*)a, (uint8_t*)dst, batch_size, ndi);
                            });
                            });
    }
    else if (K==11008){
        e = queue.submit([&](::sycl::handler& handle) {
                            handle.parallel_for(Range,
                            [=](::sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                                gemmCommonDim11008Fp16NoReshape((uint8_t*)w, (uint8_t*)a, (uint8_t*)dst, batch_size, ndi);
                            });
                            });
    }
    // end = std::chrono::high_resolution_clock::now();
    // std::cout << "sycl esimd kernel launch cost time: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " us" << std::endl;

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
        auto in_shape = params->input_layouts[0].get_shape();
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

        OPENVINO_ASSERT(out_shape.size() == 3);
        size_t M = out_shape[1];
        size_t N = out_shape[2];
        //size_t K = params->weights_layout.value().get_partial_shape()[1].get_length(); // fix 4096?
        size_t K = in_shape[2];

        void* in = static_cast<void*>(inputs[0]->buffer_ptr());
        void* wei = static_cast<void*>(weights->buffer_ptr());
        void* out = static_cast<void*>(output->buffer_ptr());


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

            std::cerr << "in = " << in << std::endl;
            std::cerr << "wei = " << wei << std::endl;
            std::cerr << "out = " << out << std::endl;
        }

	//OPENVINO_ASSERT(inputs.size() >= 2);

        const uint8_t* in_1 = static_cast<const uint8_t*>(inputs[0]->buffer_ptr());
        const uint8_t* wei_1 = static_cast<const uint8_t*>(weights->buffer_ptr());
        uint8_t* out_1 = static_cast<uint8_t*>(output->buffer_ptr());

        return to_ocl_event(stream, run_fc_fp16(sycl_queue, in_1, wei_1, out_1, M, N, K, out_t));
            
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params) {
        auto source_weights_layout = impl_params.get_input_layout(1);
        auto target_weights_layout = source_weights_layout;
        if (source_weights_layout.get_partial_shape()[1]==4096){
            target_weights_layout.format = format::os_i_osv16;
        }
        else if (source_weights_layout.get_partial_shape()[1]==11008){
            target_weights_layout.format = format::oiyx;
        }
        //target_weights_layout.format = format::oiyx;


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