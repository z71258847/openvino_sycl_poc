#include <ext/intel/esimd.hpp>
using fp16 = ::sycl::half;

using namespace ::sycl::ext::intel::esimd;
using namespace ::sycl;
using namespace ::sycl::ext::intel::esimd;
using namespace ::sycl::ext::intel::esimd::xmx;
ESIMD_INLINE void matrixMulCommonDim4096Int4NoReshape(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, nd_item<1>& ndi) {
  __ESIMD_NS::slm_init(4096 * sizeof(fp16) + 64 * sizeof(fp16));
  int hhh = ndi.get_global_id(0); // [0, 512*64)
  int hh = ndi.get_local_linear_id(); // [0, 64)
  int h = hhh >> 6; // [0, 512)
  int offsetA = (h * 8 * 4096 + hh * 64 * 8) >> 1;
  int offsetQuan = h * 64 * 16 * sizeof(fp16) + hh * 16 * sizeof(fp16);
  int offsetB = hh * 64 * sizeof(fp16);
  int outputOffset = 8 * h;
  simd<unsigned char, 256> aaa;
  simd<fp16, 16> quant;
  simd<fp16, 64> bb;
  simd<fp16, 32 * 16> aa;
  simd<fp16, 16> cc(0.0f);

  bb.template bit_cast_view<unsigned char>().template select<128, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    128,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

  aaa.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    256,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)a + offsetA);

  quant.template bit_cast_view<unsigned char>().template select<32, 1>(0) =
    __ESIMD_ENS::lsc_block_load<
    uint8_t,
    32,
    __ESIMD_ENS::lsc_data_size::default_size,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached>((uint8_t*)d + offsetQuan);

#pragma unroll
  for (int k = 0; k < 16; k++) {
    aa.select<16, 2>(32 * k) = aaa.select<16, 1>(16 * k) & 0xf;
    aa.select<16, 2>(32 * k + 1) = aaa.select<16, 1>(16 * k) >> 4;
  }

  aa = aa - 8.0f;
#pragma unroll
  for (int k = 0; k < 16; k++) {
    aa.select<32, 1>(32 * k) = quant[k] * aa.select<32, 1>(32 * k);
  }

  slm_block_store<fp16, 64>(hh * 64 * sizeof(fp16), bb);
  barrier();
  int inputVectSlmOffset = hh & 0x7;
  inputVectSlmOffset = inputVectSlmOffset * 512 * sizeof(fp16);
#pragma unroll
  for (int ll = 0; ll < 2; ll++) {
    simd<fp16, 16 * 16> fBb;
#pragma unroll
    for (int k = 0; k < 2; k++) {
      fBb.template select<128, 1>(128 * k) = slm_block_load<fp16, 128>(inputVectSlmOffset + 128 * k * sizeof(fp16));
    }

#pragma unroll
    for (int k = 0; k < 16; k++) {
      cc += aa.select<16, 1>(16 * k + ll * 256) * fBb.select<16, 1>(16 * k);
    }
    inputVectSlmOffset += 256 * sizeof(fp16);
  }

  cc.select<8, 1>(0) += cc.select<8, 1>(8);
  cc.select<4, 1>(0) += cc.select<4, 1>(4);
  cc.select<2, 1>(0) += cc.select<2, 1>(2);
  simd<fp16, 1> slmAccumulationTemp = cc[0] + cc[1];
  uint32_t slmAccumulationOffsetGroup = hh >> 3;
  uint32_t slmAccumulationOffsetSimdSlot = hh & 0x7;
  uint32_t slmAccumulationOffset = 4096 * sizeof(fp16) + (slmAccumulationOffsetGroup + slmAccumulationOffsetSimdSlot * 8) * sizeof(fp16);

  //slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
  slm_block_store<fp16, 1>(slmAccumulationOffset, slmAccumulationTemp);
  barrier();

  if (hh == 0) {
    simd<fp16, 64> sum;
    //simd<fp16, 8> sumOut;

    sum = slm_block_load<fp16, 64>(4096 * sizeof(fp16));

#pragma unroll
    for (int k = 1; k < 8; k++) {
      sum.select<8, 1>(0) += sum.select<8, 1>(8 * k);
    }

    //sumOut = sum.select<8, 1>(0);

    __ESIMD_ENS::lsc_block_store<
      fp16,
      8,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((fp16*)c + outputOffset, sum.select<8, 1>(0));
  }
}