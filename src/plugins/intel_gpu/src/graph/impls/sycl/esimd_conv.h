#include <ext/intel/esimd.hpp>
using fp16 = ::sycl::half;

using namespace ::sycl::ext::intel::esimd;
using namespace ::sycl;
using namespace ::sycl::ext::intel::esimd;
using namespace ::sycl::ext::intel::esimd::xmx;


ESIMD_INLINE void convolution7x7_3_64_s2_int8(
    uint8_t* inputBuf,
    uint8_t* weightBuf,
    uint8_t* biasBuf,
    uint8_t* scalesBuf,
    uint8_t* outputBuf,
    int width,
    int height,
    sycl::nd_item<2>& ndi)
{
    __ESIMD_NS::slm_init(64 * 7 * 8 * 3);

    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };


    uint32_t lx = ndi.get_local_id(0);
    uint32_t ly = ndi.get_local_id(1);

    uint32_t gx = ndi.get_global_range(0)-1 ;
    uint32_t gy = ndi.get_global_range(1)-1  ;

    uint32_t x = ndi.get_global_id(0);
    uint32_t y = ndi.get_global_id(1);

    simd<uint8_t, 256>  weight1;
    int slmOffset= lx * 8 + ly;
    if (slmOffset < 42) {

        weight1=__ESIMD_ENS::lsc_block_load<uint8_t,256,
            __ESIMD_ENS::lsc_data_size::u8,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)weightBuf + slmOffset * 256);

        slm_block_store<char, 256>(slmOffset * 256, weight1);
 
    }
    barrier();

    simd<uint32_t, 16> offset16(baseOffsetInc16);
    simd<int, 64 * 8>  res;    // output 4 x 4 x 64 features  64 GRF 
    simd<float, 64>     bias;   //  weight  64x1                2 GRF
    simd<float, 64>     scales;   //  weight  64x1                2 GRF
        
	bias.select<32,1>(0) = __ESIMD_ENS::lsc_block_load<float, 32,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((float*)biasBuf + 0);

    bias.select<32, 1>(32) = __ESIMD_ENS::lsc_block_load<float, 32,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((float*)biasBuf + 32);
        
    scales.select<32,1>(0) = __ESIMD_ENS::lsc_block_load<float, 32,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((float*)scalesBuf + 0);

    scales.select<32, 1>(32) = __ESIMD_ENS::lsc_block_load<float, 32,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((float*)scalesBuf + 32);

    for (int i = 0; i < 8; ++i) {
        res.select<64, 1>(i * 64 + 0) = bias;
    }

    simd<int, 16> inputOfsset1 = (y * 8 - 3) * width + (x * 4 - 4) + offset16 * width;
    simd<int, 16> inputOfsset;
    simd<uint8_t, 16 * 12> input;    

    for (int c = 0; c < 3; ++c)
    {
        inputOfsset = inputOfsset1 + c * width * height;
        inputOfsset.merge(0, inputOfsset < 0);
        input.template bit_cast_view<uint32_t>().template select<48, 1>(0) =
            __ESIMD_ENS::lsc_gather<
            uint32_t,
            3,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::none,
            __ESIMD_ENS::cache_hint::none,
            16,
            uint32_t>((uint32_t*)inputBuf, inputOfsset);

        if (x == 0 && y == 0 && c == 0)
        {
            input.select<4, 1>(64 + 12) = input.select<4, 1>();
            input.select<4, 1>(128 + 12) = input.select<4, 1>(64);
        }
        if (x == 0)
        {
            input.select<64, 1>(0) = 0;
        }
        if (x == gx) {
            input.select<64, 1>(128) = 0;
        }
        if (y == 0) {
            input.bit_cast_view<int>().select<3, 16>(0) = 0;
            input.bit_cast_view<int>().select<3, 16>(1) = 0;
            input.bit_cast_view<int>().select<3, 16>(2) = 0;
        }
        if (y == gy)
        {
            input.bit_cast_view<int>().select<3, 16>(11) = 0;
            input.bit_cast_view<int>().select<3, 16>(12) = 0;
        }

 
        for (int k = 0; k < 7; ++k)
        {
            simd < int, 64 > w1 = slm_block_load<int, 64>( (c * 7 * 8 * 64 + k * 512));
            simd < int, 64 > w2 = slm_block_load<int, 64>( (c * 7 * 8 * 64 + k * 512 + 256));
         

            w1.template bit_cast_view<char>().template select<64, 4>(0) = 0;
            
            for (int i = 0; i < 4; ++i)
            {
                int s1_0 = input.select<4, 1>(4 * k + 2 * i * 4).template bit_cast_view<int>().template select<1, 1>(0);
                int s1_1 = input.select<4, 1>(4 * k + 2 * i * 4 + 64).template bit_cast_view<int>().template select<1, 1>(0);;
                int s2_0 = input.bit_cast_view<int>().select<2, 16>(2 * i + k).bit_cast_view<char>().select<4, 1>(2).bit_cast_view<int>().select<1, 1>(0);
                int s2_1 = input.bit_cast_view<int>().select<2, 16>(2 * i + k + 16).bit_cast_view<char>().select<4, 1>(2).bit_cast_view<int>().select<1, 1>(0);
                res.select<64, 1>(i * 128)      = dp4a<int, int, int, int, 64>(res.select<64, 1>(i * 128),      w1, s1_0);
                res.select<64, 1>(i * 128)      = dp4a<int, int, int, int, 64>(res.select<64, 1>(i * 128),      w2, s1_1);
                res.select<64, 1>(i * 128 + 64) = dp4a<int, int, int, int, 64>(res.select<64, 1>(i * 128 + 64), w1, s2_0);
                res.select<64, 1>(i * 128 + 64) = dp4a<int, int, int, int, 64>(res.select<64, 1>(i * 128 + 64), w2, s2_1);
            }

        }


    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        res.select<64, 1>(i * 64 + 0).merge(0, res.select<64, 1>(i * 64 + 0) < 0);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        simd<char, 128>   res_int8;
        res_int8.select<128, 1>(0) = res.select<128, 1>(i * 128) ;
  
        __ESIMD_ENS::lsc_block_store<
            uint32_t,
            32,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back>(
                (uint32_t*)outputBuf + ((y * 4 + i) * 64 * width / 2 + (x * 2 * 64))/4, res_int8.bit_cast_view<int>());
    }

}
