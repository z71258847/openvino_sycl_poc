#include <ext/intel/esimd.hpp>
using fp16 = ::sycl::half;

using namespace ::sycl::ext::intel::esimd;
using namespace ::sycl;
using namespace ::sycl::ext::intel::esimd;
using namespace ::sycl::ext::intel::esimd::xmx;

template<uint32_t K_DIM, uint32_t pixelPerGroupShift>
ESIMD_INLINE void GEMV_Int4Weight_FP32InOutNx16Temp_largeGRF_block_ppg1_128item_per_thread(uint8_t* a, uint8_t* b, uint8_t* c, nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 1 << pixelPerGroupShift;
  constexpr uint32_t quantPerGroup = K_DIM / 32 * pixelPerGroup;
  uint32_t sumThreads = pixelPerGroup / 16;
  if (sumThreads == 0)
  {
    sumThreads = 1;
  }

  constexpr uint32_t K_DIM_DIV_4096 = (K_DIM + 4095) / 4096;
  constexpr uint32_t K_DIM_MOD_4096 = (K_DIM - (K_DIM_DIV_4096-1)*4096);
  constexpr uint32_t K_DIM_REDUCE_T = K_DIM_MOD_4096 / 128; // 16 threads, 256 ele per thread in total 4096
  __ESIMD_NS::slm_init(768 / 128 * sizeof(float));
  int hh = ndi.get_local_id(0); // [0, 64)
  int h = ndi.get_group(0); // [0, 256)
  int rowSize = ndi.get_group_range(0) * pixelPerGroup;
  int offsetABase = (h * pixelPerGroup * K_DIM + hh * 128) * sizeof(fp16);
  int offsetB = hh * 128 * sizeof(fp16);
  int outputOffset = pixelPerGroup * h;
  simd<fp16, 128> bb;
  simd<fp16, 128> aa;
  simd<fp16, 16> cc(0.0f);
  uint32_t offsetA;
  offsetA = offsetABase;

#pragma unroll
  for (int k = 0; k < K_DIM_DIV_4096; k++) {
    if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
    {
      bb.template bit_cast_view<unsigned char>().template select<128, 1>(1024 * k) =
        block_load<
        uint8_t,
        128>((uint8_t*)b + offsetB);
      bb.template bit_cast_view<unsigned char>().template select<128, 1>(1024 * k + 128*1) =
        block_load<
        uint8_t,
        128>((uint8_t*)b + offsetB + 1*64 * sizeof(fp16));  
    } 
    else 
    {
      bb.template bit_cast_view<unsigned char>().template select<128, 1>(1024 * k) = 0;
      bb.template bit_cast_view<unsigned char>().template select<128, 1>(1024 * k + 128*1) = 0;
    }

    offsetB += 4096 * sizeof(fp16);
  }

  for (int n = 0; n < pixelPerGroup; n++) {
    cc = 0.0f;

    offsetA = offsetABase + n * K_DIM * sizeof(fp16);

#pragma unroll
    for (int k = 0; k < K_DIM_DIV_4096; k++) {
      if (k != K_DIM_DIV_4096-1 || hh < K_DIM_REDUCE_T)
      {
        
        aa.template bit_cast_view<uint8_t>().template select<256, 1>(0) =
        block_load<uint8_t, 256>((uint8_t*)a + offsetA);

#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
          cc += aa.select<16, 1>(16 * kk) * bb.select<16, 1>(16 * kk + 128 * 2 * k); // note: 128 * 2 easy to have mistake!
        }
      }

      if (k == K_DIM_DIV_4096-1)
      {
        offsetA += K_DIM_REDUCE_T * 128 * sizeof(fp16);
      }
      else
      {
        offsetA += 4096 * sizeof(fp16);
      }
    }

    cc.select<8, 1>(0) += cc.select<8, 1>(8);
    cc.select<4, 1>(0) += cc.select<4, 1>(4);
    cc.select<2, 1>(0) += cc.select<2, 1>(2);
    simd<fp16, 1> slmAccumulationTemp = cc[0] + cc[1];
    uint32_t slmAccumulationOffset = hh * sizeof(fp16);
    slm_block_store<fp16, 1>(slmAccumulationOffset, slmAccumulationTemp);
  }
  barrier();
  if (hh < sumThreads) {
    {
      bb.select<8, 1>(0) = slm_block_load<fp16, 8>(0);
      bb.select<4, 1>(0) += bb.select<4, 1>(4);
      bb.select<2, 1>(0) += bb.select<2, 1>(2);

      block_store<fp16, 1>((fp16*)c + outputOffset, bb[0] + bb[1]);
      return;
    }
  }
}