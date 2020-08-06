/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_OCL_OCL_TYPES_H
#define GPU_OCL_OCL_TYPES_H

    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    #if DT_BF16 || SRC_DT_BF16 || WEI_DT_BF16 || DST_DT_BF16 || BIA_DT_BF16 \
            || A_DT_BF16 || B_DT_BF16 || C_DT_BF16 || SUM_DT_BF16
        #define MATH_UTILS_DECLARE_BF16 1
    #endif

    #define unroll_for __attribute__((opencl_unroll_hint)) for

    #define for_ for

    #define CONCAt2(a, b) a##b
    #define CONCAT2(a, b) CONCAt2(a, b)
    #define CONCAT3(a, b, c) CONCAT2(CONCAT2(a, b), c)

    #if (DT_F16 == 1) || (SRC_DT_F16 == 1) || (DST_DT_F16 == 1) \
            || (WEI_DT_F16 == 1) || (BIA_DT_F16 == 1) || (ACC_DT_F16 == 1)
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    #endif

    #if DT_F32 == 1
        #define DATA_T float
        #define DATA2_T float2
        #define DATA4_T float4
        #define DATA8_T float8
        #define DATA_MAX FLT_MAX
        #define DATA_MIN -DATA_MAX
        #define DATA_ZERO 0.0f
        #define DATA_ONE 1.0f
        #define DEF_ACC_DATA_T float
        #define DEF_ACC_DATA2_T float2
        #define DEF_ACC_DATA4_T float4
        #define DEF_ACC_DATA8_T float8
        #define POST_OP_DATA_T float
        #define TO_DATA_T(v) (float)(v)
        #define TO_DEF_ACC_DATA_T(v) (float)(v)
        #define DATA_TO_REF convert_float
        #define CONVERT_DATA_T convert_float
        #define CONVERT_DATA2_T convert_float2
        #define CONVERT_DATA4_T convert_float4
        #define CONVERT_DATA8_T convert_float8
        #define CONVERT_FLOAT_T convert_float
        #define CONVERT_FLOAT2_T convert_float2
        #define CONVERT_FLOAT4_T convert_float4
        #define CONVERT_FLOAT8_T convert_float8

        #define BLOCK_READ intel_sub_group_block_read
        #define BLOCK_WRITE intel_sub_group_block_write
        #define BLOCK_READ2 intel_sub_group_block_read2
        #define BLOCK_READ4 intel_sub_group_block_read4
        #define BLOCK_READ8 intel_sub_group_block_read8
        #define BLOCK_WRITE2 intel_sub_group_block_write2
        #define BLOCK_WRITE4 intel_sub_group_block_write4
        #define BLOCK_WRITE8 intel_sub_group_block_write8

        #define AS_DATA_T as_float
        #define AS_DATA2_T as_float2
        #define AS_DATA4_T as_float4
        #define AS_DATA8_T as_float8

        #define AS_UINT_T as_uint
        #define AS_UINT2_T as_uint2
        #define AS_UINT4_T as_uint4
        #define AS_UINT8_T as_uint8

        #define BLOCK_DATA_T uint
        #define BLOCK_DATA2_T uint2
        #define BLOCK_DATA4_T uint4
        #define BLOCK_DATA8_T uint8
        #define AS_BLOCK_DATA_T as_uint
        #define AS_BLOCK_DATA2_T as_uint2
        #define AS_BLOCK_DATA4_T as_uint4
        #define AS_BLOCK_DATA8_T as_uint8
    #elif DT_F16 == 1

        #define DATA_T half
        #define DATA2_T half2
        #define DATA4_T half4
        #define DATA8_T half8
        #define DATA16_T half16
        #define AS_DATA2_T as_half2
        #define DATA_MAX HALF_MAX
        #define DATA_MIN -DATA_MAX
        #define DATA_ZERO 0.0h
        #define DATA_ONE 1.0h
        #define DEF_ACC_DATA_T half
        #define DEF_ACC_DATA2_T half2
        #define DEF_ACC_DATA4_T half4
        #define DEF_ACC_DATA8_T half8
        #define POST_OP_DATA_T half
        #define TO_DATA_T(v) (half)(v)
        #define TO_DEF_ACC_DATA_T(v) (half)(v)
        #define DATA_TO_REF convert_half
        #define CONVERT_DATA_T convert_half
        #define CONVERT_DATA2_T convert_half2
        #define CONVERT_DATA4_T convert_half4
        #define CONVERT_DATA8_T convert_half8
        #define CONVERT_FLOAT_T convert_float
        #define CONVERT_FLOAT2_T convert_float2
        #define CONVERT_FLOAT4_T convert_float4
        #define CONVERT_FLOAT8_T convert_float8

        #define BLOCK_READ intel_sub_group_block_read_us
        #define BLOCK_WRITE intel_sub_group_block_write_us
        #define BLOCK_READ2 intel_sub_group_block_read_us2
        #define BLOCK_READ4 intel_sub_group_block_read_us4
        #define BLOCK_READ8 intel_sub_group_block_read_us8
        #define BLOCK_WRITE2 intel_sub_group_block_write_us2
        #define BLOCK_WRITE4 intel_sub_group_block_write_us4
        #define BLOCK_WRITE8 intel_sub_group_block_write_us8
        #define AS_DATA_T as_half
        #define AS_DATA2_T as_half2
        #define AS_DATA4_T as_half4
        #define AS_DATA8_T as_half8

        #define AS_UINT_T as_ushort
        #define AS_UINT2_T as_ushort2
        #define AS_UINT4_T as_ushort4
        #define AS_UINT8_T as_ushort8

        #define BLOCK_DATA_T ushort
        #define BLOCK_DATA2_T ushort2
        #define BLOCK_DATA4_T ushort4
        #define BLOCK_DATA8_T ushort8
        #define AS_BLOCK_DATA_T as_ushort
        #define AS_BLOCK_DATA2_T as_ushort2
        #define AS_BLOCK_DATA4_T as_ushort4
        #define AS_BLOCK_DATA8_T as_ushort8

        #define MMAD_DATA_T uint
        #define MMAD_DATA4_T uint4
        #define MMAD_DATA8_T uint8
        #define MMAD_ACC_DATA4_T float4
        #define MMAD_ACC_DATA8_T float8
    #elif DT_BF16 == 1
        #define DATA_T ushort
        #define DATA2_T ushort2
        #define POST_OP_DATA_T float
        #define DATA2_T ushort2
        #define DATA4_T ushort4
        #define DATA8_T ushort8
        #define DATA16_T ushort16
        #define DATA_MAX as_float(0x7f7f0000)
        #define DATA_MIN (-DATA_MAX)
        #define DATA_ZERO 0.0f
        #define DATA_ONE 1.0f
        #define DEF_ACC_DATA_T float
        #define DEF_ACC_DATA2_T float2
        #define DEF_ACC_DATA4_T float4
        #define DEF_ACC_DATA8_T float8
        #define TO_DATA_T(v) cvt_f32_to_bf16(v)
        #define TO_DEF_ACC_DATA_T(v) cvt_bf16_to_f32(v)
        #define DATA_TO_REF cvt_bf16_to_f32
        #define CONVERT_DATA_T cvt_f32_to_bf16
        #define CONVERT_DATA2_T cvt_f32_to_bf16
        #define CONVERT_DATA4_T cvt_f32_to_bf16
        #define CONVERT_DATA8_T cvt_f32_to_bf16
        #define CONVERT_FLOAT_T cvt_bf16_to_f32
        #define CONVERT_FLOAT2_T cvt_bf16_to_f32
        #define CONVERT_FLOAT4_T cvt_bf16_to_f32
        #define CONVERT_FLOAT8_T cvt_bf16_to_f32

        #define BLOCK_READ intel_sub_group_block_read_us
        #define BLOCK_WRITE intel_sub_group_block_write_us
        #define BLOCK_READ2 intel_sub_group_block_read_us2
        #define BLOCK_READ4 intel_sub_group_block_read_us4
        #define BLOCK_READ8 intel_sub_group_block_read_us8
        #define BLOCK_WRITE2 intel_sub_group_block_write_us2
        #define BLOCK_WRITE4 intel_sub_group_block_write_us4
        #define BLOCK_WRITE8 intel_sub_group_block_write_us8
        #define AS_DATA_T as_ushort
        #define AS_DATA2_T as_ushort2
        #define AS_DATA4_T as_ushort4
        #define AS_DATA8_T as_ushort8

        #define AS_UINT_T as_ushort
        #define AS_UINT2_T as_ushort2
        #define AS_UINT4_T as_ushort4
        #define AS_UINT8_T as_ushort8

        #define BLOCK_DATA_T ushort
        #define BLOCK_DATA2_T ushort2
        #define BLOCK_DATA4_T ushort4
        #define BLOCK_DATA8_T ushort8
        #define AS_BLOCK_DATA_T as_ushort
        #define AS_BLOCK_DATA2_T as_ushort2
        #define AS_BLOCK_DATA4_T as_ushort4
        #define AS_BLOCK_DATA8_T as_ushort8

        #define MMAD_DATA_T uint
        #define MMAD_DATA4_T uint4
        #define MMAD_DATA8_T uint8
        #define MMAD_ACC_DATA4_T float4
        #define MMAD_ACC_DATA8_T float8
    #elif DT_S8 == 1
        #define DATA_T char
        #define DATA2_T char2
        #define DATA4_T char4
        #define DATA8_T char8
        #define DATA16_T char16
        #define DATA_MAX CHAR_MAX
        #define DATA_MIN CHAR_MIN
        #define DATA_ZERO 0
        #define DATA_ONE 1
        #define INT8_T int8
        #define DEF_ACC_DATA_T int
        #define DEF_ACC_DATA2_T int2
        #define DEF_ACC_DATA4_T int4
        #define DEF_ACC_DATA8_T int8
        #define POST_OP_DATA_T float
        #define TO_DATA_T(v) convert_char_sat_rte(v)
        #define TO_DEF_ACC_DATA_T(v) (float)(v)
        #define DATA_TO_REF convert_float
        #define CONVERT_DATA_T convert_char_sat_rte
        #define CONVERT_DATA2_T convert_char2_sat_rte
        #define CONVERT_DATA4_T convert_char4_sat_rte
        #define CONVERT_DATA8_T convert_char8_sat_rte
        #define CONVERT_FLOAT_T convert_float
        #define CONVERT_FLOAT2_T convert_float2
        #define CONVERT_FLOAT4_T convert_float4
        #define CONVERT_FLOAT8_T convert_float8

        #define BLOCK_READ intel_sub_group_block_read_uc
        #define BLOCK_WRITE intel_sub_group_block_write_uc
        #define BLOCK_READ2 intel_sub_group_block_read_uc2
        #define BLOCK_READ4 intel_sub_group_block_read_uc4
        #define BLOCK_READ8 intel_sub_group_block_read_uc8
        #define BLOCK_WRITE2 intel_sub_group_block_write_uc2
        #define BLOCK_WRITE4 intel_sub_group_block_write_uc4
        #define BLOCK_WRITE8 intel_sub_group_block_write_uc8
        #define AS_DATA_T as_char
        #define AS_DATA2_T as_char2
        #define AS_DATA4_T as_char4
        #define AS_DATA8_T as_char8
        #define AS_DATA16_T as_char16

        #define AS_UINT_T as_uchar
        #define AS_UINT2_T as_uchar2
        #define AS_UINT4_T as_uchar4
        #define AS_UINT8_T as_uchar8
        #define AS_INT8_T as_int8

        #define BLOCK_DATA_T uchar
        #define BLOCK_DATA2_T uchar2
        #define BLOCK_DATA4_T uchar4
        #define BLOCK_DATA8_T uchar8
        #define AS_BLOCK_DATA_T as_uchar
        #define AS_BLOCK_DATA2_T as_uchar2
        #define AS_BLOCK_DATA4_T as_uchar4
        #define AS_BLOCK_DATA8_T as_uchar8

        #define MMAD_DATA_T int
        #define MMAD_DATA4_T int4
        #define MMAD_DATA8_T int8
        #define MMAD_ACC_DATA4_T int4
        #define MMAD_ACC_DATA8_T int8
    #elif DT_U8 == 1
        #define DATA_T uchar
        #define DATA2_T uchar2
        #define DATA4_T uchar4
        #define DATA8_T uchar8
        #define DATA16_T uchar16
        #define DATA_MAX UCHAR_MAX
        #define DATA_MIN 0
        #define DATA_ZERO 0
        #define DATA_ONE 1
        #define INT8_T uint8
        #define DEF_ACC_DATA_T int
        #define DEF_ACC_DATA2_T int2
        #define DEF_ACC_DATA4_T int4
        #define DEF_ACC_DATA8_T int8
        #define POST_OP_DATA_T float
        #define TO_DATA_T(v) convert_uchar_sat_rte(v)
        #define TO_DEF_ACC_DATA_T(v) (float)(v)
        #define DATA_TO_REF convert_float
        #define CONVERT_DATA_T convert_uchar_sat_rte
        #define CONVERT_DATA2_T convert_uchar2_sat_rte
        #define CONVERT_DATA4_T convert_uchar4_sat_rte
        #define CONVERT_DATA8_T convert_uchar8_sat_rte
        #define CONVERT_FLOAT_T convert_float
        #define CONVERT_FLOAT2_T convert_float2
        #define CONVERT_FLOAT4_T convert_float4
        #define CONVERT_FLOAT8_T convert_float8

        #define BLOCK_READ intel_sub_group_block_read_uc
        #define BLOCK_WRITE intel_sub_group_block_write_uc
        #define BLOCK_READ2 intel_sub_group_block_read_uc2
        #define BLOCK_READ4 intel_sub_group_block_read_uc4
        #define BLOCK_READ8 intel_sub_group_block_read_uc8
        #define BLOCK_WRITE2 intel_sub_group_block_write_uc2
        #define BLOCK_WRITE4 intel_sub_group_block_write_uc4
        #define BLOCK_WRITE8 intel_sub_group_block_write_uc8
        #define AS_DATA_T as_uchar
        #define AS_DATA2_T as_uchar2
        #define AS_DATA4_T as_uchar4
        #define AS_DATA8_T as_uchar8
        #define AS_DATA16_T as_uchar16

        #define AS_UINT_T as_uchar
        #define AS_UINT2_T as_uchar2
        #define AS_UINT4_T as_uchar4
        #define AS_UINT8_T as_uchar8
        #define AS_INT8_T as_uint8

        #define BLOCK_DATA_T uchar
        #define BLOCK_DATA2_T uchar2
        #define BLOCK_DATA4_T uchar4
        #define BLOCK_DATA8_T uchar8
        #define AS_BLOCK_DATA_T as_uchar
        #define AS_BLOCK_DATA2_T as_uchar2
        #define AS_BLOCK_DATA4_T as_uchar4
        #define AS_BLOCK_DATA8_T as_uchar8

        #define MMAD_DATA_T uint
        #define MMAD_DATA4_T uint4
        #define MMAD_DATA8_T uint8
        #define MMAD_ACC_DATA4_T int4
        #define MMAD_ACC_DATA8_T int8
    #elif DT_S32 == 1
        #define DATA_T int
        #define DATA_TO_REF convert_float
        #define CONVERT_DATA_T convert_int_sat_rte
        #define POST_OP_DATA_T float
    #elif !defined(DT_UNDEF)
        #error "Unexpected data type"
    #endif

    #if VECT_DT_N == 1
        #define VECT_DATA_T DATA_T
        #define VECT_DEF_ACC_DATA_T DEF_ACC_DATA_T
        #define AS_VECT_DATA_T AS_DATA_T
        #define VECT_BLOCK_READ BLOCK_READ
        #define VECT_BLOCK_WRITE BLOCK_WRITE
        #define VECT_UINT_READ intel_sub_group_block_read
        #define VECT_UINT_WRITE intel_sub_group_block_write
        #define VECT_BLOCK_DATA_T BLOCK_DATA_T
        #define AS_VECT_BLOCK_DATA_T AS_BLOCK_DATA_T
        #define CONVERT_VECT_FLOAT_T CONVERT_FLOAT_T
        #define CONVERT_VECTOR_DATA_T CONVERT_DATA_T
        #define VECT_INT_T int
        #define VECT_UINT_T uint
        #define VECT_FLOAT_T float
        #define AS_VECT_INT_T as_int
        #define AS_VECT_UINT_T as_uint
        #define AS_VECT_FLOAT_T as_float
    #elif VECT_DT_N == 2
        #define VECT_DATA_T DATA2_T
        #define VECT_DEF_ACC_DATA_T DEF_ACC_DATA2_T
        #define AS_VECT_DATA_T AS_DATA2_T
        #define VECT_BLOCK_READ BLOCK_READ2
        #define VECT_BLOCK_WRITE BLOCK_WRITE2
        #define VECT_UINT_READ intel_sub_group_block_read2
        #define VECT_UINT_WRITE intel_sub_group_block_write2
        #define VECT_BLOCK_DATA_T BLOCK_DATA2_T
        #define AS_VECT_BLOCK_DATA_T AS_BLOCK_DATA2_T
        #define CONVERT_VECT_FLOAT_T CONVERT_FLOAT2_T
        #define CONVERT_VECTOR_DATA_T CONVERT_DATA2_T
        #define VECT_INT_T int2
        #define VECT_UINT_T uint2
        #define VECT_FLOAT_T float2
        #define AS_VECT_INT_T as_int2
        #define AS_VECT_UINT_T as_uint2
        #define AS_VECT_FLOAT_T as_float2
    #elif VECT_DT_N == 4
        #define VECT_DATA_T DATA4_T
        #define VECT_DEF_ACC_DATA_T DEF_ACC_DATA4_T
        #define AS_VECT_DATA_T AS_DATA4_T
        #define VECT_BLOCK_READ BLOCK_READ4
        #define VECT_BLOCK_WRITE BLOCK_WRITE4
        #define VECT_UINT_READ intel_sub_group_block_read4
        #define VECT_UINT_WRITE intel_sub_group_block_write4
        #define VECT_BLOCK_DATA_T BLOCK_DATA4_T
        #define AS_VECT_BLOCK_DATA_T AS_BLOCK_DATA4_T
        #define CONVERT_VECT_FLOAT_T CONVERT_FLOAT4_T
        #define CONVERT_VECTOR_DATA_T CONVERT_DATA4_T
        #define VECT_INT_T int4
        #define VECT_UINT_T uint4
        #define VECT_FLOAT_T float4
        #define AS_VECT_INT_T as_int4
        #define AS_VECT_UINT_T as_uint4
        #define AS_VECT_FLOAT_T as_float4
    #elif VECT_DT_N == 8
        #define VECT_DATA_T DATA8_T
        #define VECT_DEF_ACC_DATA_T DEF_ACC_DATA8_T
        #define AS_VECT_DATA_T AS_DATA8_T
        #define VECT_BLOCK_READ BLOCK_READ8
        #define VECT_BLOCK_WRITE BLOCK_WRITE8
        #define VECT_UINT_READ intel_sub_group_block_read8
        #define VECT_UINT_WRITE intel_sub_group_block_write8
        #define VECT_BLOCK_DATA_T BLOCK_DATA8_T
        #define AS_VECT_BLOCK_DATA_T AS_BLOCK_DATA8_T
        #define CONVERT_VECT_FLOAT_T CONVERT_FLOAT8_T
        #define CONVERT_VECTOR_DATA_T CONVERT_DATA8_T
        #define VECT_INT_T int8
        #define VECT_UINT_T uint8
        #define VECT_FLOAT_T float8
        #define AS_VECT_INT_T as_int8
        #define AS_VECT_UINT_T as_uint8
        #define AS_VECT_FLOAT_T as_float8
    #endif

    #ifdef SRC_DATA_T
        #define SRC_DATA2_T CONCAT2(SRC_DATA_T, 2)
        #define SRC_DATA4_T CONCAT2(SRC_DATA_T, 4)
        #define SRC_DATA8_T CONCAT2(SRC_DATA_T, 8)
        #define SRC_DATA16_T CONCAT2(SRC_DATA_T, 16)
        #ifdef SRC_DT_S8
            #define SRC_MMAD_DATA_T int
            #define SRC_MMAD_DATA4_T int4
            #define SRC_MMAD_DATA8_T int8
        #else
            #define SRC_MMAD_DATA_T uint
            #define SRC_MMAD_DATA4_T uint4
            #define SRC_MMAD_DATA8_T uint8
        #endif

        #if defined(SRC_DT_U8) || defined(SRC_DT_S8)
            #define SRC_MMAD_ACC_DATA4_T int4
            #define SRC_MMAD_ACC_DATA8_T int8
        #else
            #define SRC_MMAD_ACC_DATA4_T float4
            #define SRC_MMAD_ACC_DATA8_T float8
        #endif


        #if SRC_DT_BF16
            #define SRC_TO_REF(x) cvt_bf16_to_f32(x)
            #define SRC_TO_REF8(x) cvt_bf16_to_f32(x)
            #define REF_TO_SRC(x) cvt_f32_to_bf16(x)
        #else
            #define SRC_TO_REF(x) (x)
            #define SRC_TO_REF8(x) (x)
            #define REF_TO_SRC(x) (x)
        #endif

        #if SRC_DT_BF16
            #define TO_SRC(x) cvt_f32_to_bf16(x)
        #elif SRC_DT_U8
            #define TO_SRC(x) convert_uchar_sat_rte(x)
        #elif SRC_DT_S8
            #define TO_SRC(x) convert_char_sat_rte(x)
        #elif SRC_DT_S32
            #define TO_SRC(x) convert_int_sat_rte(x)
        #else
            #define TO_SRC(x) (x)
        #endif
        #ifdef SRC_DT_S8
            #define SRC_MMAD_DATA_T int
            #define SRC_MMAD_DATA4_T int4
            #define SRC_MMAD_DATA8_T int8
        #else
            #define SRC_MMAD_DATA_T uint
            #define SRC_MMAD_DATA4_T uint4
            #define SRC_MMAD_DATA8_T uint8
        #endif

        #if defined(SRC_DT_U8) || defined(SRC_DT_S8)
            #define SRC_MMAD_ACC_DATA4_T int4
            #define SRC_MMAD_ACC_DATA8_T int8
        #else
            #define SRC_MMAD_ACC_DATA4_T float4
            #define SRC_MMAD_ACC_DATA8_T float8
        #endif


        #if SRC_DT_BF16
            #define SRC_TO_REF(x) cvt_bf16_to_f32(x)
            #define SRC_TO_REF8(x) cvt_bf16_to_f32(x)
            #define REF_TO_SRC(x) cvt_f32_to_bf16(x)
        #else
            #define SRC_TO_REF(x) (x)
            #define SRC_TO_REF8(x) (x)
            #define REF_TO_SRC(x) (x)
        #endif

        #if SRC_DT_BF16
            #define TO_SRC(x) cvt_f32_to_bf16(x)
        #elif SRC_DT_U8
            #define TO_SRC(x) convert_uchar_sat_rte(x)
        #elif SRC_DT_S8
            #define TO_SRC(x) convert_char_sat_rte(x)
        #elif SRC_DT_S32
            #define TO_SRC(x) convert_int_sat_rte(x)
        #else
            #define TO_SRC(x) (x)
        #endif
    #endif

    #ifdef A_DATA_T
        #define A_DATA8_T CONCAT2(A_DATA_T, 8)
        #if A_DT_BF16
            #define A_TO_REF(x) cvt_bf16_to_f32(x)
            #define A_TO_REF8(x) cvt_bf16_to_f32(x)
            #define REF_TO_A(x) cvt_f32_to_bf16(x)
        #else
            #define A_TO_REF(x) (x)
            #define A_TO_REF8(x) (x)
            #define REF_TO_A(x) (x)
        #endif
        #if A_DT_BF16
            #define TO_A(x) cvt_f32_to_bf16(x)
        #elif A_DT_U8
            #define TO_A(x) convert_uchar_sat_rte(x)
        #elif A_DT_S8
            #define TO_A(x) convert_char_sat_rte(x)
        #elif A_DT_S32
            #define TO_A(x) convert_int_sat_rte(x)
        #else
            #define TO_A(x) (x)
        #endif
    #endif

    #ifdef WEI_DATA_T
        #if WEI_DT_BF16
        #define WEI_TO_REF(x) cvt_bf16_to_f32(x)
        #define REF_TO_WEI(x) cvt_f32_to_bf16(x)
        #else
        #define WEI_TO_REF(x) (x)
        #define REF_TO_WEI(x) (x)
        #endif
        #if WEI_DT_BF16
        #define TO_WEI(x) cvt_f32_to_bf16(x)
        #elif WEI_DT_U8
        #define TO_WEI(x) convert_uchar_sat_rte(x)
        #elif WEI_DT_S8
        #define TO_WEI(x) convert_char_sat_rte(x)
        #elif WEI_DT_S32
        #define TO_WEI(x) convert_int_sat_rte(x)
        #else
        #define TO_WEI(x) (x)
        #endif
    #endif

    #ifdef B_DATA_T
        #if B_DT_BF16
        #define B_TO_REF(x) cvt_bf16_to_f32(x)
        #define REF_TO_B(x) cvt_f32_to_bf16(x)
        #else
        #define B_TO_REF(x) (x)
        #define REF_TO_B(x) (x)
        #endif
        #if B_DT_BF16
        #define TO_B(x) cvt_f32_to_bf16(x)
        #elif B_DT_U8
        #define TO_B(x) convert_uchar_sat_rte(x)
        #elif B_DT_S8
        #define TO_B(x) convert_char_sat_rte(x)
        #elif B_DT_S32
        #define TO_B(x) convert_int_sat_rte(x)
        #else
        #define TO_B(x) (x)
        #endif
    #endif

    #ifdef BIA_DATA_T
        #define BIA_DATA2_T CONCAT2(BIA_DATA_T, 2)
        #if BIA_DT_BF16
        #define BIA_TO_REF(x) cvt_bf16_to_f32(x)
        #define REF_TO_BIA(x) cvt_f32_to_bf16(x)
        #else
        #define BIA_TO_REF(x) (x)
        #define REF_TO_BIA(x) (x)
        #endif
        #if BIA_DT_BF16
        #define TO_BIA(x) cvt_f32_to_bf16(x)
        #elif BIA_DT_U8
        #define TO_BIA(x) convert_uchar_sat_rte(x)
        #elif BIA_DT_S8
        #define TO_BIA(x) convert_char_sat_rte(x)
        #elif BIA_DT_S32
        #define TO_BIA(x) convert_int_sat_rte(x)
        #else
        #define TO_BIA(x) (x)
        #endif
    #endif

    #ifdef DST_DATA_T
        #define DST_DATA2_T CONCAT2(DST_DATA_T, 2)
        #define DST_DATA4_T CONCAT2(DST_DATA_T, 4)
        #define DST_DATA8_T CONCAT2(DST_DATA_T, 8)
        #define DST_DATA16_T CONCAT2(DST_DATA_T, 16)

        #define AS_DST_DATA2_T CONCAT2(as_, DST_DATA2_T)
        #define AS_DST_DATA4_T CONCAT2(as_, DST_DATA4_T)
        #define AS_DST_DATA8_T CONCAT2(as_, DST_DATA8_T)
        #define AS_DST_DATA16_T CONCAT2(as_, DST_DATA16_T)

        #if DST_DT_F32 || DST_DT_F16
        #define CONVERT_DST_DATA2_T CONCAT2(convert_, DST_DATA2_T)
        #define CONVERT_DST_DATA4_T CONCAT2(convert_, DST_DATA4_T)
        #define CONVERT_DST_DATA8_T CONCAT2(convert_, DST_DATA8_T)
        #define CONVERT_DST_DATA16_T CONCAT2(convert_, DST_DATA16_T)
        #else
        #define CONVERT_DST_DATA2_T CONCAT3(convert_, DST_DATA2_T, _sat_rte)
        #define CONVERT_DST_DATA4_T CONCAT3(convert_, DST_DATA4_T, _sat_rte)
        #define CONVERT_DST_DATA8_T CONCAT3(convert_, DST_DATA8_T, _sat_rte)
        #define CONVERT_DST_DATA16_T CONCAT3(convert_, DST_DATA16_T, _sat_rte)
        #endif

        // Block read/write macros for dst.
        #if DST_DT_U8 || DST_DT_S8
        #define BLOCK_READ_DST2(ptr) \
            AS_DST_DATA2_T(intel_sub_group_block_read_uc2((__global uchar *)ptr))
        #define BLOCK_WRITE_DST2(ptr, v) \
            intel_sub_group_block_write_uc2((__global uchar *)ptr, as_uchar2(v))

        #define BLOCK_READ_DST4(ptr) \
            AS_DST_DATA4_T(intel_sub_group_block_read_uc4((__global uchar *)ptr))
        #define BLOCK_WRITE_DST4(ptr, v) \
            intel_sub_group_block_write_uc4((__global uchar *)ptr, as_uchar4(v))

        #define BLOCK_READ_DST8(ptr) \
            AS_DST_DATA8_T(intel_sub_group_block_read_uc8((__global uchar *)ptr))
        #define BLOCK_WRITE_DST8(ptr, v) \
            intel_sub_group_block_write_uc8((__global uchar *)ptr, as_uchar8(v))

        #define BLOCK_READ_DST16(ptr) \
            AS_DST_DATA16_T(intel_sub_group_block_read_uc16((__global uchar *)ptr))
        #define BLOCK_WRITE_DST16(ptr, v) \
            intel_sub_group_block_write_uc16((__global uchar *)ptr, as_uchar16(v))

        #elif DST_DT_F16 || DST_DT_BF16

        #define BLOCK_READ_DST2(ptr) \
            AS_DST_DATA2_T(intel_sub_group_block_read_us2((__global ushort *)ptr))
        #define BLOCK_WRITE_DST2(ptr, v) \
            intel_sub_group_block_write_us2((__global ushort *)ptr, as_ushort2(v))

        #define BLOCK_READ_DST4(ptr) \
            AS_DST_DATA4_T(intel_sub_group_block_read_us4((__global ushort *)ptr))
        #define BLOCK_WRITE_DST4(ptr, v) \
            intel_sub_group_block_write_us4((__global ushort *)ptr, as_ushort4(v))

        #define BLOCK_READ_DST8(ptr) \
            AS_DST_DATA8_T(intel_sub_group_block_read_us8((__global ushort *)ptr))
        #define BLOCK_WRITE_DST8(ptr, v) \
            intel_sub_group_block_write_us8((__global ushort *)ptr, as_ushort8(v))

        #define BLOCK_READ_DST16(ptr) \
            (DST_DATA16_T)( \
                    BLOCK_READ_DST8(ptr), BLOCK_READ_DST8(ptr + 8 * SUB_GROUP_SIZE))
        #define BLOCK_WRITE_DST16(ptr, v) \
            do { \
                BLOCK_WRITE_DST8(ptr, (v).s01234567); \
                BLOCK_WRITE_DST8(ptr + 8 * SUB_GROUP_SIZE, (v).s89abcdef); \
            } while (0)

        #elif DST_DT_S32 || DST_DT_F32

        #define BLOCK_READ_DST2(ptr) \
            AS_DST_DATA2_T(intel_sub_group_block_read2((__global uint *)ptr))
        #define BLOCK_WRITE_DST2(ptr, v) \
            intel_sub_group_block_write2((__global uint *)ptr, as_uint2(v))

        #define BLOCK_READ_DST4(ptr) \
            AS_DST_DATA4_T(intel_sub_group_block_read4((__global uint *)ptr))
        #define BLOCK_WRITE_DST4(ptr, v) \
            intel_sub_group_block_write4((__global uint *)ptr, as_uint4(v))

        #define BLOCK_READ_DST8(ptr) \
            AS_DST_DATA8_T(intel_sub_group_block_read8((__global uint *)ptr))
        #define BLOCK_WRITE_DST8(ptr, v) \
            intel_sub_group_block_write8((__global uint *)ptr, as_uint8(v))

        #define BLOCK_READ_DST16(ptr) \
            (DST_DATA16_T)( \
                    BLOCK_READ_DST8(ptr), BLOCK_READ_DST8(ptr + 8 * SUB_GROUP_SIZE))
        #define BLOCK_WRITE_DST16(ptr, v) \
            do { \
                BLOCK_WRITE_DST8(ptr, (v).s01234567); \
                BLOCK_WRITE_DST8(ptr + 8 * SUB_GROUP_SIZE, (v).s89abcdef); \
            } while (0)

        #endif

        #if DST_DT_BF16
            #define DST_TO_REF(x) cvt_bf16_to_f32(x)
            #define DST_TO_REF2(x) cvt_bf16_to_f32(x)
            #define DST_TO_REF8(x) cvt_bf16_to_f32(x)
            #define REF_TO_DST(x) cvt_f32_to_bf16(x)
            #define REF_TO_DST8(x) cvt_f32_to_bf16(convert_float8(x))
        #elif DST_DT_F16
            #define REF_TO_DST(x) convert_half(x)
            #define DST_TO_REF(x) convert_float(x)
            #define DST_TO_REF2(x) convert_float2(x)
            #define DST_TO_REF8(x) convert_float8(x)
        #elif DST_DT_U8
            #define DST_TO_REF(x) (x)
            #define DST_TO_REF2(x) (x)
            #define DST_TO_REF8(x) (x)
            #define REF_TO_DST(x) convert_uchar(x)
            #define REF_TO_DST8(x) convert_uchar8(x)
        #elif DST_DT_S8
            #define DST_TO_REF(x) (x)
            #define DST_TO_REF2(x) (x)
            #define DST_TO_REF8(x) (x)
            #define REF_TO_DST(x) convert_char(x)
            #define REF_TO_DST8(x) convert_char8(x)
        #else
            #define DST_TO_REF(x) (x)
            #define DST_TO_REF2(x) (x)
            #define DST_TO_REF8(x) (x)
            #define REF_TO_DST(x) (x)
            #define REF_TO_DST8(x) (x)
        #endif

        #if DST_DT_BF16
            #define TO_DST(x) cvt_f32_to_bf16(x)
            #define TO_DST2(x) cvt_f32_to_bf16(convert_float2(x))
            #define TO_DST4(x) cvt_f32_to_bf16(convert_float4(x))
            #define TO_DST8(x) cvt_f32_to_bf16(convert_float8(x))
        #elif DST_DT_F16
            #define TO_DST(x) convert_half(x)
            #define TO_DST2(x) convert_half2(x)
            #define TO_DST4(x) convert_half4(x)
            #define TO_DST8(x) convert_half8(x)
        #elif DST_DT_U8
            #define TO_DST(x) convert_uchar_sat_rte(x)
            #define TO_DST2(x) convert_uchar2_sat_rte(x)
            #define TO_DST4(x) convert_uchar4_sat_rte(x)
            #define TO_DST8(x) convert_uchar8_sat_rte(x)
            #define TO_DST16(x) convert_uchar16_sat_rte(x)
        #elif DST_DT_S8
            #define TO_DST(x) convert_char_sat_rte(x)
            #define TO_DST2(x) convert_char2_sat_rte(x)
            #define TO_DST4(x) convert_char4_sat_rte(x)
            #define TO_DST8(x) convert_char8_sat_rte(x)
            #define TO_DST16(x) convert_char16_sat_rte(x)
        #elif DST_DT_S32
            #define TO_DST(x) convert_int_sat_rte(x)
            #define TO_DST2(x) convert_int2_sat_rte(x)
            #define TO_DST4(x) convert_int4_sat_rte(x)
            #define TO_DST8(x) convert_int8_sat_rte(x)
        #elif DST_DT_F32
            #define TO_DST(x) convert_float(x)
            #define TO_DST2(x) convert_float2(x)
            #define TO_DST4(x) convert_float4(x)
            #define TO_DST8(x) convert_float8(x)
        #else
            #error "Not expected"
        #endif
    #endif

    #ifdef C_DATA_T
        #define C_DATA8_T CONCAT2(C_DATA_T, 8)
        #if C_DT_BF16
            #define C_TO_REF(x) cvt_bf16_to_f32(x)
            #define C_TO_REF8(x) cvt_bf16_to_f32(x)
            #define REF_TO_C(x) cvt_f32_to_bf16(x)
            #define REF_TO_C8(x) cvt_f32_to_bf16(convert_float8(x))
        #else
            #define C_TO_REF(x) (x)
            #define C_TO_REF8(x) (x)
            #define REF_TO_C(x) (x)
            #define REF_TO_C8(x) (x)
        #endif

        #if C_DT_BF16
            #define TO_C(x) cvt_f32_to_bf16(x)
            #define TO_C8(x) cvt_f32_to_bf16(convert_float8(x))
        #elif C_DT_F16
            #define TO_C(x) convert_half(x)
            #define TO_C8(x) convert_half8(x)
        #elif C_DT_U8
            #define TO_C(x) convert_uchar_sat_rte(x)
            #define TO_C8(x) convert_uchar8_sat_rte(x)
        #elif C_DT_S8
            #define TO_C(x) convert_char_sat_rte(x)
            #define TO_C8(x) convert_char8_sat_rte(x)
        #elif C_DT_S32
            #define TO_C(x) convert_int_sat_rte(x)
            #define TO_C8(x) convert_int8_sat_rte(x)
        #elif C_DT_F32
            #define TO_C(x) convert_float(x)
            #define TO_C8(x) convert_float8(x)
        #else
            #error "Not expected"
        #endif
    #endif

    #ifdef ACC_DATA_T
        #if ACC_DT_F16
            #define TO_ACC(x) convert_half(x)
        #elif ACC_DT_F32
            #define TO_ACC(x) convert_float(x)
        #elif ACC_DT_S32
            #define TO_ACC(x) convert_int(x)
        #else
            #error "Unexpected accumulation data type"
        #endif
    #endif

    #ifdef SUM_DATA_T
        #define SUM_DATA2_T CONCAT2(SUM_DATA_T, 2)
        #define SUM_DATA4_T CONCAT2(SUM_DATA_T, 4)
        #define SUM_DATA8_T CONCAT2(SUM_DATA_T, 8)
        #define SUM_DATA16_T CONCAT2(SUM_DATA_T, 16)
        #define AS_SUM_DATA_T CONCAT2(as_, SUM_DATA_T)
        #define AS_SUM_DATA2_T CONCAT2(as_, SUM_DATA2_T)
        #define AS_SUM_DATA4_T CONCAT2(as_, SUM_DATA4_T)
        #define AS_SUM_DATA8_T CONCAT2(as_, SUM_DATA8_T)
        #define AS_SUM_DATA16_T CONCAT2(as_, SUM_DATA16_T)
        #if SUM_DT_BF16
            #define SUM_TO_REF cvt_bf16_to_f32
        #else
            #define SUM_TO_REF
        #endif
    #endif


#endif
