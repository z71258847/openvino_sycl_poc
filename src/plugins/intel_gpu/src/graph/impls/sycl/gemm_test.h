#include <ext/intel/esimd.hpp>
using fp16 = ::sycl::half;

using namespace ::sycl::ext::intel::esimd;
using namespace ::sycl;
using namespace ::sycl::ext::intel::esimd;
using namespace ::sycl::ext::intel::esimd::xmx;
#define XMX_USED

#ifdef GEMM_FP16AW_CAL_REF
template<uint32_t REDUCE_K_DIM_DIV_32>
ESIMD_INLINE void gemmReduce2048WeightsFP16InputShffuledFp16_xmx_ppifull_bb8_notmp_ctile(uint8_t* a, uint8_t* b, uint8_t* c, 
int outputRow, int hiddenDim, int tokenSize, nd_item<2>& ndi) {
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    constexpr uint32_t baseOffsetInc4[4] = { 0, 1, 2, 3 };
    constexpr uint32_t loopCount = 8;  // 256 / 8 / 4 = 8
    constexpr uint32_t loopCountW = 2;  // 256 / 16 / 8 = 2
    int hh = ndi.get_local_linear_id(); // [0, 64)
    int h = ndi.get_group(0); // [0, (row + 15) / 16)
    int v = 0; // [0, (row + 15) / 16)
    int hiddenDimFP16Size = hiddenDim * sizeof(fp16);
    simd<fp16, 8 * 16 * 2 * 8> cc;
    simd<fp16, 8 * 32 * 8> bb;
    simd<fp16, 32 * 16 * 2> aa;
    simd<fp16, 32 * 16> aat;
    simd<fp16, 8* 16 * 2> dd;
    
    simd<fp16, 16> aaa = (fp16)(-8.0f);  
    simd<uint8_t, 16 * 32> bitShift;
    simd<uint8_t, 16 * 32> bitShiftTemp;
    // WG:
    // 4096 / 256 = 16    1024 / 256 = 4 
    int tokenBlkCnt = (tokenSize + 255) / 256;
    int h_i = h % tokenBlkCnt;   // 4
    int h_w = h / tokenBlkCnt;   // 16

    int hi = h_i * 32/* 256/8 */ + (hh >> 3) * 8;   // 4 t for input and each do 8 * 8

    int hh_8 = (hh & 0x7);

    if (hi * 8 >= tokenSize) return;

    int hw222 = h_w * 16/* 256/16 */ + hh_8 * 2 + 0;  // nw is 0, read 2 weights
    // handle weight non-common dim not 256 aligned, assume 128 aligned
    if (outputRow % 256 != 0 && outputRow % 128 == 0 && h_w > 0)
    {
      hw222 -= 8;
    }
    cc = 0;

    uint32_t globalOffset = v * 2048 + hw222 * hiddenDim * 16;
    uint32_t baseOffsetA = globalOffset * sizeof(fp16);

    simd<uint32_t, 16> offset(baseOffsetInc16);
    simd<uint32_t, 16> offsetQuant(baseOffsetInc16);

    offset = offset * hiddenDimFP16Size + baseOffsetA;

    uint32_t baseOffsetB = hi * 8*32*sizeof(fp16)*REDUCE_K_DIM_DIV_32;

    #pragma unroll
    for (int ww = 0; ww < REDUCE_K_DIM_DIV_32 /*2048/32*/ ; ww++) 
    {

    
    {

    {
        aa.template bit_cast_view<uint32_t>().template select<128, 1>(0) =
        __ESIMD_ENS::lsc_gather<
        uint32_t,
        8,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        16,
        uint32_t
        >((uint32_t*)a, offset + ww * 32 * sizeof(fp16));
        aa.template bit_cast_view<uint32_t>().template select<128, 1>(128) =
        __ESIMD_ENS::lsc_gather<
        uint32_t,
        8,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        16,
        uint32_t
        >((uint32_t*)a, offset + ww * 32 * sizeof(fp16) + 16 * sizeof(fp16));

        aa.template bit_cast_view<uint32_t>().template select<128, 1>(256) =
        __ESIMD_ENS::lsc_gather<
        uint32_t,
        8,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        16,
        uint32_t
        >((uint32_t*)a, offset + hiddenDimFP16Size * 16 + ww * 32 * sizeof(fp16));

        aa.template bit_cast_view<uint32_t>().template select<128, 1>(384) =
        __ESIMD_ENS::lsc_gather<
        uint32_t,
        8,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        16,
        uint32_t
        >((uint32_t*)a, offset + hiddenDimFP16Size * 16 + ww * 32 * sizeof(fp16) + 16 * sizeof(fp16));

    }

    } // www

    // barrier();

    #pragma unroll
    for (int nn = 0; nn < loopCount; nn++)
    {

    {
        {
        //read 256 fp16 =8*8*4.
        bb.template bit_cast_view<uint8_t>().template select<256, 1>(/*www*8*32*2 +*/nn*8*32*2 + 0) =
            __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB + nn * 256*sizeof(fp16)*REDUCE_K_DIM_DIV_32 + ww * 256 * sizeof(fp16));
        bb.template bit_cast_view<uint8_t>().template select<256, 1>(/*www*8*32*2 +*/nn*8*32*2 + 256) =
            __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB + 256 + nn * 256*sizeof(fp16)*REDUCE_K_DIM_DIV_32 + ww * 256 * sizeof(fp16));
        }
    } // www
    } // loop i


    #pragma unroll
    for (int nn = 0; nn < loopCount; nn++)
    {
    
    #pragma unroll
    for (int nw = 0; nw < loopCountW; nw++)
    {
        aat.select<512, 1>(0) = aa.select<512, 1>(nw * 512);

    {
        {
            
        simd<sycl::half, 8 * 16> bb_tmp{0};
        simd<sycl::half, 8 * 16> bb_xmx{0};
        simd<sycl::half, 16 * 16> aa_xmx{0};
        simd<sycl::half, 8 * 16> cc_xmx{0};

        cc_xmx.select<16*8,1>(0)=cc.select<16*8,1>((nw*8 + nn) * 16*8);
        //bb_xmx=bb_tmp.select<8*16,1>(0);
        bb_xmx=bb.select<8*16,1>(/*www*8*32*/nn*8*32 +0);
    #pragma unroll
        for(int t=0;t<16;t++)//t<N/2
        {
            aa_xmx.template select<16,1>(16 * t)=aat.template  select<16,1>(16*t);
        }
    #ifdef XMX_USED
        cc_xmx = xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(cc_xmx, aa_xmx, bb_xmx);
    #endif

        bb_xmx=bb.select<8*16,1>(/*www*8*32*/nn*8*32 + 8*16);
    #pragma unroll
        for(int t=0;t<16;t++)//t<N/2
        {
            aa_xmx.template select<16,1>(16 * t)=aat.template  select<16,1>(16*t + 8*16*2*1);
        }
    #ifdef XMX_USED
        cc_xmx = xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(cc_xmx, aa_xmx, bb_xmx);
    #endif

        cc.select<16*8,1>((nw*8 + nn) * 16*8)=cc_xmx.select<16*8,1>(0);

        }
        
    } // www


    } // loop w
    } // loop i

    } // loop common w

    #pragma unroll
    for (int nn = 0; nn < loopCount; nn++)
    {
    if (hi * 8 + nn * 8 + 7 < tokenSize)
    {
    #pragma unroll
    for (int nw = 0; nw < loopCountW; nw++)
    {
        int hw = h_w * 16/* 256/16 */ + (hh & 0x7) * 2 + nw;  // 8t for w and each do 2 * 16 
        // handle weight non-common dim not 256 aligned, assume 128 aligned
        if (outputRow % 256 != 0 && outputRow % 128 == 0 && h_w > 0)
        {
          hw -= 8;
        }
        uint32_t offsetC = hw * 16 + nn * 8 * outputRow + hi * 8 * outputRow;  // hi include 

    #pragma unroll
        for (int k = 0; k < 8; k++) {
        __ESIMD_ENS::lsc_block_store<
            fp16,
            16,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>((fp16*)c + offsetC + k * outputRow, cc.select<16,1>((nw*8 + nn) * 16*8 + 16 * k));
        }
    } // loop w
    }
    else
    {
    #pragma unroll
    for (int nw = 0; nw < loopCountW; nw++)
    {
        int hw = h_w * 16/* 256/16 */ + (hh & 0x7) * 2 + nw;  // 8t for w and each do 2 * 16 
        // handle weight non-common dim not 256 aligned, assume 128 aligned
        if (outputRow % 256 != 0 && outputRow % 128 == 0 && h_w > 0)
        {
          hw -= 8;
        }
        uint32_t offsetC = hw * 16 + nn * 8 * outputRow + hi * 8 * outputRow;  // hi include 

    #pragma unroll
        for (int k = 0; k < 8; k++) {
        if (hi * 8 + nn * 8 + k < tokenSize) {
        __ESIMD_ENS::lsc_block_store<
            fp16,
            16,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>((fp16*)c + offsetC + k * outputRow, cc.select<16,1>((nw*8 + nn) * 16*8 + 16 * k));
        }
        }
    } // loop w
    }
    } // loop i

}


ESIMD_INLINE void fp16ShuffleToFp16_xmx_no_k_split_ref(uint8_t* a, uint8_t* b, uint32_t hiddenDim, uint32_t tokenLength, nd_item<2>& ndi) {
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  int h = ndi.get_group(0); // [0, n // 32)
  int v = ndi.get_group(1); // [0, aligned(k, 8) // 8)
  uint32_t baseOffset = (v * 8 * hiddenDim + h * 32) * sizeof(fp16);
  int offsetOut = (v * hiddenDim / 32 + h) * 8 * 32;

  simd<uint32_t, 8> offsetIn(baseOffsetInc8);
  simd<fp16, 256> fp16Input;
  simd<fp16, 256> fp16Output;
  offsetIn = offsetIn * hiddenDim * sizeof(fp16) + baseOffset;

  simd_mask<8> quantPred = 1;
  if (v*8 + 7 >= tokenLength)
  {
    quantPred = 0;
    int mask = tokenLength - v*8;
    for (int t = 0; t < mask; t++) {
      quantPred[t] = 1;
    }
  }

#pragma unroll
  for (int k = 0; k < 4 * 8; k++) {
    fp16Input.template bit_cast_view<fp16>().template select<8, 1>(8 * k) =
      __ESIMD_ENS::lsc_gather<
      fp16,
      1,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      8,
      uint32_t
      >((fp16*)a, offsetIn, quantPred);
    offsetIn += sizeof(fp16);
  }

// 2x8x16
#pragma unroll
  for (int k = 0; k < 2; k++) {
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
        fp16Output.select<16, 1>(k*128 + kk*16) = fp16Input.select<16, 8>(k*128 + kk);
    }
  }


#pragma unroll
  for (int k = 0; k < 2; k++) {
    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)b + offsetOut + 128 * k, fp16Output.select<128, 1>(128 * k));
  }
}

#endif

ESIMD_INLINE void fp16ShuffleToFp16_xmx_no_k_split(uint8_t* a, uint8_t* b, uint32_t hiddenDim, uint32_t tokenLength, nd_item<2>& ndi) {
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  int h = ndi.get_group(0); // [0, n // 32)
  int v = ndi.get_group(1); // [0, aligned(k, 8) // 8)
  uint32_t baseOffset = (v * 8 * hiddenDim + h * 32) * sizeof(fp16);
  int offsetOut = (v * hiddenDim / 32 + h) * 8 * 32;

  simd<uint32_t, 8> offsetIn(baseOffsetInc8);
  simd<fp16, 256> fp16Input;
  simd<fp16, 256> fp16Output;
  offsetIn = offsetIn * hiddenDim * sizeof(fp16) + baseOffset;

  simd_mask<8> quantPred = 1;
  if (v*8 + 7 >= tokenLength)
  {
    quantPred = 0;
    int mask = tokenLength - v*8;
    for (int t = 0; t < mask; t++) {
      quantPred[t] = 1;
    }
  }

#pragma unroll
  for (int k = 0; k < 4 * 8; k++) {
    fp16Input.template bit_cast_view<fp16>().template select<8, 1>(8 * k) =
      __ESIMD_ENS::lsc_gather<
      fp16,
      1,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      8,
      uint32_t
      >((fp16*)a, offsetIn, quantPred);
    offsetIn += sizeof(fp16);
  }


// 8x32
#pragma unroll
for (int kk = 0; kk < 8; kk++) {
    fp16Output.select<32, 1>(kk*32) = fp16Input.select<32, 8>(kk);
}

#pragma unroll
  for (int k = 0; k < 2; k++) {
    __ESIMD_ENS::lsc_block_store<
      fp16,
      128,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)b + offsetOut + 128 * k, fp16Output.select<128, 1>(128 * k));
  }
}

#define INPUT_QUANT
#define WEIGHT_QUANT

#define INPUT_QUANT_BIT 8
#define WEIGHT_QUANT_BIT 8

template<uint32_t REDUCE_K_DIM_DIV_32>
ESIMD_INLINE void gemmReduce2048WeightsFP16InputShffuledFp16_xmx_ppifull_bb8_notmp_int8cal_ctile(uint8_t* a, uint8_t* b, uint8_t* c, 
int outputRow, int hiddenDim, int tokenSize, nd_item<2>& ndi) {
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t baseOffsetInc4[4] = { 0, 1, 2, 3 };
  constexpr uint32_t loopCount = 8;  // 256 / 8 / 4 = 8
  constexpr uint32_t loopCountW = 2;  // 256 / 16 / 8 = 2

  //__ESIMD_NS::slm_init(2 * 16 * 256 * sizeof(fp16));// + 8 * 16 * 2 * 8 * 4 * 8 * sizeof(fp16));   // = 256 * 256 = 65536
  int hh = ndi.get_local_linear_id(); // [0, 64)
  int h = ndi.get_group(0); // [0, (row + 15) / 16)
  int v = 0; // [0, (row + 15) / 16)

  int hiddenDimFP16Size = hiddenDim * sizeof(fp16);
  simd<fp16, 8 * 16 * 2 * 8> cc;
  simd<fp16, 8 * 32 * 8> bb;
  simd<fp16, 32 * 16 * 2> aa;
  simd<fp16, 32 * 16 * 2> aa_shuffle;
  simd<fp16, 32 * 16> aat;
  simd<float, 8* 16 * 2> dd;
  simd<float, 8 * 16> ccf;
  simd<float, 8* 16> ddb;
  
  simd<fp16, 16> aaa = (fp16)(-8.0f);  
  simd<uint8_t, 16 * 32> bitShift;
  simd<uint8_t, 16 * 32> bitShiftTemp;
  // int hw = h * 8 + (hh & 0x7); 
  // int hi = hh >> 3;

  // WG:
  // 4096 / 256 = 16    1024 / 256 = 4 
  int tokenBlkCnt = (tokenSize + 255) / 256;
  int h_i = h % tokenBlkCnt;   // 4
  int h_w = h / tokenBlkCnt;   // 16

  int hi = h_i * 32/* 256/8 */ + (hh >> 3) * 8;   // 4 t for input and each do 8 * 8

  int hh_8 = (hh & 0x7);

  if (hi * 8 >= tokenSize) return;

  int hw222 = h_w * 16/* 256/16 */ + hh_8 * 2 + 0;  // nw is 0, read 2 weights
  // handle weight non-common dim not 256 aligned, assume 128 aligned
  if (outputRow % 256 != 0 && outputRow % 128 == 0 && h_w > 0)
  {
    hw222 -= 8;
  }
  cc = 0;

  uint32_t globalOffset = v * 2048 + hw222 * hiddenDim * 16;
  uint32_t baseOffsetA = globalOffset * sizeof(fp16);

  uint32_t offset = baseOffsetA;

  uint32_t baseOffsetB = hi * 8*32*sizeof(fp16)*REDUCE_K_DIM_DIV_32;

#pragma unroll
for (int ww = 0; ww < REDUCE_K_DIM_DIV_32 /*2048/32*/ ; ww++) 
{

{
  #pragma unroll
  for (int kk = 0; kk < 32; kk++)
  {
    aa.template bit_cast_view<uint8_t>().template select<64, 1>(64 * kk) =
      __ESIMD_ENS::lsc_block_load<
      uint8_t,
      64,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offset + hiddenDimFP16Size * kk + ww * 32 * sizeof(fp16));
  }

#ifdef WEIGHT_QUANT
  simd<fp16, 32> amax_a{0};
  simd<fp16, 32> absaa{0};

  simd<fp16, 32> dda32{0};
  simd<fp16, 32* 32> dda32shuffle{0};

#pragma unroll
  for (int kk = 0; kk < 32; kk++)
  {
    absaa = abs<fp16, 32>(aa.select<32, 32>(kk));
    amax_a = max<fp16, 32, fp16>(absaa, amax_a);
  }

  dda32.select<32, 1>(0) = amax_a / ((1 << (WEIGHT_QUANT_BIT-1)) - 1);//127;//((1 << 7) - 1);
  
  dd.select<16, 1>(0) = dda32.select<16, 1>(0);  // from dda32.select<32, 1>(0)
  dd.select<16, 1>(128) = dda32.select<16, 1>(16);


#pragma unroll
  for (int kk = 0; kk < 32; kk++)
  {
    dda32shuffle.select<32, 32>(kk) = dda32.select<32, 1>(0);
  }
  
  aa = aa / dda32shuffle;
#endif // WEIGHT_QUANT

#pragma unroll
  for (int k = 0; k < 2; k++)
  {
  #pragma unroll
    for (int kk = 0; kk < 32; kk++)
    {
      aa_shuffle.select<16, 1>(k * 32*16 + kk * 16) = aa.select<16, 32>(k * 32*16 + kk);
    }
  }
  aa = aa_shuffle;


   dd.select<16, 1>(16) = dd.select<16, 1>(0);
   dd.select<32, 1>(32) = dd.select<32, 1>(0);
   dd.select<64, 1>(64) = dd.select<64, 1>(0);

   dd.select<16, 1>(128 + 16) = dd.select<16, 1>(128);
   dd.select<32, 1>(128 + 32) = dd.select<32, 1>(128);
   dd.select<64, 1>(128 + 64) = dd.select<64, 1>(128);

#ifndef WEIGHT_QUANT
    dd = 1;
#endif
} // www

#pragma unroll
for (int nn = 0; nn < loopCount; nn++)
{


{
    //if (hi * 8 + nn * 8 < tokenSize)
    {
      //read 256 fp16 =8*8*4.
      bb.template bit_cast_view<uint8_t>().template select<256, 1>(/*www*8*32*2 +*/nn*8*32*2 + 0) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB + nn * 256*sizeof(fp16)*REDUCE_K_DIM_DIV_32 + ww * 256 * sizeof(fp16));
      bb.template bit_cast_view<uint8_t>().template select<256, 1>(/*www*8*32*2 +*/nn*8*32*2 + 256) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB + 256 + nn * 256*sizeof(fp16)*REDUCE_K_DIM_DIV_32 + ww * 256 * sizeof(fp16));
    }
} // www
} // loop i


#pragma unroll
for (int nn = 0; nn < loopCount; nn++)
{
  simd<int8_t, 8 * 2 * 16> bbi_xmx{0};
#ifdef INPUT_QUANT
  simd<fp16, 8> amax{0};
  simd<fp16, 8> absbb{0};

  simd<fp16, 8* 32> ddb32{0};

#pragma unroll
  for (int kk = 0; kk < 32; kk++)
  {
    absbb = abs<fp16, 8>(bb.select<8, 32>(nn*8*32 + kk));
    amax = max<fp16, 8, fp16>(absbb, amax);
  }

  ddb.select<8, 16>(0) = amax / ((1 << (INPUT_QUANT_BIT-1)) - 1);//127;//((1 << 7) - 1);
#pragma unroll
  for (int kk = 0; kk < 16; kk++)
  {
    ddb.select<8, 16>(kk) = ddb.select<8, 16>(0);
  }
  for (int kk = 0; kk < 32; kk++)
  {
    ddb32.select<8, 32>(kk) = ddb.select<8, 16>(0);
  }
  bb.select<8*32, 1>(nn*8*32) = bb.select<8*32, 1>(nn*8*32) / ddb32;
#else
  ddb = 1;
#endif
  bbi_xmx = bb.select<8*32,1>(/*www*8*32*/nn*8*32);

#pragma unroll
for (int nw = 0; nw < loopCountW; nw++)
{

      aat.select<512, 1>(0) = aa.select<512, 1>(nw * 512);

// INT8 XMX cal
{
    {
      simd<int8_t, 16 * 2 * 16> aai_xmx{0};
      simd<int32_t, 8 * 16> cci_xmx{0};

      //cci_xmx.select<16*8,1>(0)=cc.select<16*8,1>((nw*8 + nn) * 16*8);
  // #pragma unroll
  //     for(int t=0;t<8;t++)
  //     {
  //       bbi_xmx.select<16,1>(t * 32)=bb.select<16,1>(/*www*8*32*/nn*8*32 +0 + t*16);
  //       bbi_xmx.select<16,1>(t * 32 + 16)=bb.select<16,1>(/*www*8*32*/nn*8*32 + 8*16 + t*16);
  //     }
  #pragma unroll
      for(int t=0;t<32;t++)//t<N/2
      {
        aai_xmx.template select<16,4>(t%4 + 64*(t/4))=aat.template  select<16,1>(16*t);
      }
#ifdef XMX_USED
      cci_xmx = xmx::dpas<8, 8, int32_t, int32_t, int8_t, int8_t>(cci_xmx, aai_xmx, bbi_xmx);
#endif
      ccf = cci_xmx;
      cc.select<16*8,1>((nw*8 + nn) * 16*8) += ccf.select<16*8,1>(0) * dd.select<16*8,1>(16*8 * nw) * ddb;
      // cc.select<16*8,1>((nw*8 + nn) * 16*8) = cc.select<16*8,1>((nw*8 + nn) * 16*8) * dd.select<16*8,1>(0);
    }
} // INT8 XMX cal

} // loop w
} // loop i

} // loop common w

#pragma unroll
for (int nn = 0; nn < loopCount; nn++)
{
if (hi * 8 + nn * 8 + 7 < tokenSize)
{
#pragma unroll
for (int nw = 0; nw < loopCountW; nw++)
{
    int hw = h_w * 16/* 256/16 */ + (hh & 0x7) * 2 + nw;  // 8t for w and each do 2 * 16 
    // handle weight non-common dim not 256 aligned, assume 128 aligned
    if (outputRow % 256 != 0 && outputRow % 128 == 0 && h_w > 0)
    {
      hw -= 8;
    }
    uint32_t offsetC = hw * 16 + nn * 8 * outputRow + hi * 8 * outputRow;  // hi include 

  #pragma unroll
     for (int k = 0; k < 8; k++) {
      __ESIMD_ENS::lsc_block_store<
        fp16,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>((fp16*)c + offsetC + k * outputRow, cc.select<16,1>((nw*8 + nn) * 16*8 + 16 * k));
     }
} // loop w
}
else
{
#pragma unroll
for (int nw = 0; nw < loopCountW; nw++)
{
    int hw = h_w * 16/* 256/16 */ + (hh & 0x7) * 2 + nw;  // 8t for w and each do 2 * 16 
    // handle weight non-common dim not 256 aligned, assume 128 aligned
    if (outputRow % 256 != 0 && outputRow % 128 == 0 && h_w > 0)
    {
      hw -= 8;
    }
    uint32_t offsetC = hw * 16 + nn * 8 * outputRow + hi * 8 * outputRow;  // hi include 

  #pragma unroll
     for (int k = 0; k < 8; k++) {
      if (hi * 8 + nn * 8 + k < tokenSize) {
      __ESIMD_ENS::lsc_block_store<
        fp16,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>((fp16*)c + offsetC + k * outputRow, cc.select<16,1>((nw*8 + nn) * 16*8 + 16 * k));
      }
     }
} // loop w
}
} // loop i

}
