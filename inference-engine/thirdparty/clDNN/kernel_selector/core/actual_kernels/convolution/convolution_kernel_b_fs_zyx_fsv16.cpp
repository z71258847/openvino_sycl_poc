//
// Copyright (c) 2019-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "convolution_kernel_b_fs_zyx_fsv16.h"
#include "kernel_selector_utils.h"
#include <algorithm>
#include <iostream>
#include <string>

namespace kernel_selector {

static const size_t sub_group_size = 16;
static const size_t feature_block_size = 16;

struct conf_t {
    int mb;
    int ngroups, ic, oc;
    int ic_without_padding;
    int oc_without_padding;
    int id, ih, iw, od, oh, ow;
    int kd, kh, kw;
    int stride_d, stride_h, stride_w;
    int dilate_d, dilate_h, dilate_w;

    int f_pad, l_pad, t_pad;
    int back_pad, r_pad, b_pad;

    int od_block, oh_block, ow_block;
    int id_block, ih_block, iw_block;
    int oc_block, ic_block, nchunk;
    int omb;
    int odb, ohb, owb;
    int icb;
    int ocb;
    int mb_block;
    int sub_group_size;

    bool is_depthwise;
    bool is_nhwc;
    bool ver_1stconv;
    bool ver_16mb16c;
    bool is_nchw;
    bool is_src_nchw, is_src_nhwc;
};

template <typename T, typename U>
inline typename std::remove_reference<T>::type rnd_dn(const T a, const U b) {
    return static_cast<typename std::remove_reference<T>::type>((a / b) * b);
}

template <typename T, typename U>
inline typename std::remove_reference<T>::type max_div(const T a, const U b) {
    U div = b;
    while (div > 1) {
        if (a % div == 0) return div;
        div--;
    }
    return static_cast<typename std::remove_reference<T>::type>(div);
}

static void fwd_compute_block_sizes(conf_t &conf, const convolution_params& params) {
    const auto& out = params.output;
    const auto& input = params.inputs[0];
    const auto& weights = params.weights;

    conf.oc = out.Feature().v;
    conf.od = out.Z().v;
    conf.oh = out.Y().v;
    conf.ow = out.X().v;

    conf.ic = input.Feature().v;
    conf.id = input.Z().v;
    conf.ih = input.Y().v;
    conf.iw = input.X().v;

    conf.dilate_d = 0;//params.dilation.z - 1;
    conf.dilate_h = params.dilation.y - 1;
    conf.dilate_w = params.dilation.x - 1;

    conf.f_pad = 0;//params.padding.z;
    conf.t_pad = params.padding.y;
    conf.l_pad = params.padding.x;

    conf.back_pad = 0;//params.padding.z;
    conf.b_pad = params.padding.y;
    conf.r_pad = params.padding.x;

    conf.stride_d = 1;//params.padding.z;
    conf.stride_h = params.padding.y;
    conf.stride_w = params.padding.x;

    conf.kd = 1;//weights.Z().v;
    conf.kh = weights.Y().v;
    conf.kw = weights.X().v;

    conf.ic_block = 1;
    conf.od_block = 1;
    conf.oh_block = 1;
    conf.ow_block = 1;

    conf.mb = out.Batch().v;
    conf.ngroups = params.groups;
    conf.ic_without_padding = input.Feature().v;
    conf.oc_without_padding = out.Feature().v;

    conf.ver_1stconv = input.Feature().v == 3;
    conf.is_depthwise = input.Feature().v == (size_t)conf.ngroups;
    conf.ver_16mb16c = !conf.ver_1stconv &&
        ((out.GetDType() == Datatype::F16 && conf.mb % 32 == 0) ||
        (out.GetDType() == Datatype::F32 && conf.mb % 16 == 0));

    int max_ow_block = (out.GetDType() == Datatype::F16 ? 20 : 16);
    if (conf.ver_16mb16c) {
        max_ow_block = 1;
    } else if (conf.is_depthwise || conf.ver_1stconv) {
        max_ow_block = 8;
    }
    max_ow_block = std::min(conf.ow, max_ow_block);

    if (conf.ver_16mb16c) {
        conf.mb_block = (out.GetDType() == Datatype::F16) ? (conf.mb % 32 == 0 ? 32 : 16) : 16;
    } else {
        conf.mb_block = 1;
    }
    conf.ow_block = max_div(conf.ow, max_ow_block);

    if (conf.ow_block < max_ow_block / 2) {
        float min_tail_ratio = 1;
        int best_ow_block = -1;
        for (int ow_block = 8; ow_block <= max_ow_block; ow_block++) {
            float tail_ratio
                    = (ow_block - (conf.ow % ow_block)) / (float)conf.ow;
            if (tail_ratio <= min_tail_ratio) {
                min_tail_ratio = tail_ratio;
                best_ow_block = ow_block;
            }
        }
        assert(best_ow_block > 0);
        conf.ow_block = best_ow_block;
    }

    if (conf.is_depthwise) {
        conf.oc_block = 16;
        conf.ic_block = 16;
        conf.omb = conf.mb_block;
        return;
    }

    if (conf.ver_1stconv && conf.mb_block == 1 && conf.oc % 32 == 0) {
        conf.oc_block = 32;
    } else {
        conf.oc_block = 16;
    }
    conf.ic_block = std::min(conf.ic, 16);

    conf.omb = (conf.mb_block == 1 && conf.mb % 16 == 0) ? 16 : conf.mb_block;
    conf.ocb = max_div(conf.oc / 16, 8) * 16;
}

FusedOpsConfiguration GenerateFusedOpsConfiguration_f16(size_t conf_id, std::string input_name, Datatype dt,
                                                        bool is_vector) {
    std::vector<std::string> idx_order;
    std::string suffix = (is_vector ? "_VEC" : "_SCALAR") + std::to_string(conf_id);
    std::string input_var_name = input_name + std::to_string(conf_id) + (is_vector ? "" : "[i]");
    size_t vec_size = is_vector ? 8 : 1;
    if (is_vector)
        idx_order = {"(mb)", "(oc*OC_BLOCK + g*OC)", "od", "oh", "(ow + " + std::to_string(conf_id * 8) + ")"};
    else
        idx_order = {"(mb)", "(oc*OC_BLOCK + g*OC + local_id)", "od", "oh", "(ow + " + std::to_string(conf_id * 8) + " + i)"};

    return { suffix,
             idx_order,
             input_var_name,
             dt,
             vec_size,
             is_vector ? FusedOpsConfiguration::LoadType::LT_ALIGNED_READ : FusedOpsConfiguration::LoadType::LT_UNALIGNED,
             FusedOpsConfiguration::BoundaryCheck::ENABLED,
             FusedOpsConfiguration::IndexType::TENSOR_COORD,
             Tensor::DataChannelName::X };
}

FusedOpsConfiguration GenerateFusedOpsConfiguration_bsv16_fsv16(size_t conf_id, std::string input_name, Datatype dt,
                                                                size_t dims) {
    std::vector<std::string> idx_order;
    if (dims == 5)
        idx_order = {"(mb + " + std::to_string(conf_id * 8) + ")", "(oc*16)", "od", "oh", "ow"};
    else
        idx_order = {"(mb + " + std::to_string(conf_id * 8) + ")", "(oc*16)", "oh", "ow"};

    return { "_VEC" + std::to_string(conf_id),
             idx_order,
             input_name + std::to_string(conf_id),
             dt,
             8,
             FusedOpsConfiguration::LoadType::LT_ALIGNED_READ,
             FusedOpsConfiguration::BoundaryCheck::ENABLED,
             FusedOpsConfiguration::IndexType::TENSOR_COORD,
             Tensor::DataChannelName::BATCH };
}

ParamsKey ConvolutionKernel_b_fs_zyx_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableSplitSupport();
    k.EnableBatching();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    k.EnableGroupedConvolution();
    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_b_fs_zyx_fsv16::SetDefault(const convolution_params& params,
                                                                           int autoTuneIndex) const {
    DispatchData kd = ConvolutionKernelBase::SetDefault(params, autoTuneIndex);

    conf_t conf;
    fwd_compute_block_sizes(conf, params);

    kd.gws0 = conf.ngroups * conf.ocb / (conf.oc_block / sub_group_size);
    kd.gws1 = conf.od * conf.oh * CeilDiv(conf.ow, conf.ow_block) * (conf.omb / conf.mb_block);
    kd.gws2 = (conf.oc / conf.ocb) * (conf.mb / conf.omb);

    kd.lws0 = sub_group_size;
    kd.lws1 = 1;
    kd.lws2 = 1;
    if (conf.mb == 1)
        kd.efficiency = FORCE_PRIORITY_2;
    else
        kd.efficiency = FORCE_PRIORITY_7;

    return kd;
}

bool ConvolutionKernel_b_fs_zyx_fsv16::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o) || !CovolutionCheckInput(p, o)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

    // if (output.Feature().v % feature_block_size != 0)
    //     return false;

    if (input.GetLayout() == DataLayout::bfzyx || input.GetLayout() == DataLayout::bfyx) {
        if (input.Feature().v != 3)
            return false;
        if (output.GetDType() == Datatype::F16 && (output.Feature().v % 32 != 0))
            return false;
    } else {
        // if ((input.Feature().v / params.groups) % feature_block_size != 0 && (input.Feature().v / params.groups) != 8)
        //     return false;
    }

    // Check that padding before features doesn't miss-align the blocks
    if (input.Feature().pad.before % feature_block_size != 0 || output.Feature().pad.before % feature_block_size != 0) {
        return false;
    }

    return true;
}

JitConstants ConvolutionKernel_b_fs_zyx_fsv16::GetJitConstants(const convolution_params& params,
                                                         const DispatchData& runInfo) const {
    auto input = params.inputs[0];
    auto output = params.output;
    auto jit = Parent::GetJitConstants(params, runInfo);

    conf_t conf;
    fwd_compute_block_sizes(conf, params);

    if (conf.ver_16mb16c) {
        jit.AddConstant(MakeJitConstant("VER_16MB16C", 1));
    } else {
        jit.AddConstant(MakeJitConstant("VER_8OW16C", 1));
    }

    auto input_dt = GetUnitType(params);

    jit.AddConstant(MakeJitConstant("DATA_T", toCLType(input_dt)));

    switch(input_dt) {
        case Datatype::F16:  {
            jit.AddConstant(MakeJitConstant("DT_F16", 1));
            break;
        }
        case Datatype::F32:  {
            jit.AddConstant(MakeJitConstant("DT_F32", 1));
            break;
        }
        default: break;
    }

    jit.AddConstant(MakeJitConstant("SRC_W16C", !conf.ver_1stconv));
    jit.AddConstant(MakeJitConstant("SRC_16N16C", 0));
    jit.AddConstant(MakeJitConstant("SRC_NHWC", 0));
    jit.AddConstant(MakeJitConstant("SRC_NCHW", conf.ver_1stconv));

    jit.AddConstant(MakeJitConstant("WEI_I16O", 0));
    jit.AddConstant(MakeJitConstant("WEI_16I16O", 1));
    jit.AddConstant(MakeJitConstant("WEI_16I16O_FLIPPED", 0));

    jit.AddConstant(MakeJitConstant("DST_W16C", 1));
    jit.AddConstant(MakeJitConstant("DST_16N16C", 0));

    if (conf.ver_16mb16c && !params.fused_ops.empty()) {
        const auto dims_num = DataTensor::ChannelsCount(input.GetLayout());
        if (output.GetDType() != Datatype::F16) {
            FusedOpsConfiguration conf_vec0 = GenerateFusedOpsConfiguration_bsv16_fsv16(0, "blockC0", input_dt, dims_num);
            FusedOpsConfiguration conf_vec1 = GenerateFusedOpsConfiguration_bsv16_fsv16(1, "blockC0", input_dt, dims_num);
            jit.Merge(MakeFusedOpsJitConstants(params, {conf_vec0, conf_vec1}));
        } else {
            FusedOpsConfiguration conf_vec0 = GenerateFusedOpsConfiguration_bsv16_fsv16(0, "C0", input_dt, dims_num);
            FusedOpsConfiguration conf_vec1 = GenerateFusedOpsConfiguration_bsv16_fsv16(1, "C0", input_dt, dims_num);
            FusedOpsConfiguration conf_vec2 = GenerateFusedOpsConfiguration_bsv16_fsv16(2, "C0", input_dt, dims_num);
            FusedOpsConfiguration conf_vec3 = GenerateFusedOpsConfiguration_bsv16_fsv16(3, "C0", input_dt, dims_num);
            jit.Merge(MakeFusedOpsJitConstants(params, {conf_vec0, conf_vec1, conf_vec2, conf_vec3}));
        }
    } else if (!conf.ver_1stconv && !params.fused_ops.empty()) {
        FusedOpsConfiguration conf_vec0 = GenerateFusedOpsConfiguration_f16(0, "blockC0", input_dt, true);
        FusedOpsConfiguration conf_vec1 = GenerateFusedOpsConfiguration_f16(1, "blockC0", input_dt, true);
        FusedOpsConfiguration conf_scalar0 = GenerateFusedOpsConfiguration_f16(0, "blockC0", input_dt, false);
        FusedOpsConfiguration conf_scalar1 = GenerateFusedOpsConfiguration_f16(1, "blockC0", input_dt, false);
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_vec0, conf_vec1, conf_scalar0, conf_scalar1}));
    }

    jit.AddConstant(MakeJitConstant("LWS_0", runInfo.lws0));
    jit.AddConstant(MakeJitConstant("LWS_1", runInfo.lws1));
    jit.AddConstant(MakeJitConstant("LWS_2", runInfo.lws2));


    jit.AddConstant(MakeJitConstant("DD", conf.dilate_d));
    jit.AddConstant(MakeJitConstant("DH", conf.dilate_h));
    jit.AddConstant(MakeJitConstant("DW", conf.dilate_w));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("FWD_DATA", 1));
    jit.AddConstant(MakeJitConstant("IS_DW", conf.is_depthwise));
    jit.AddConstant(MakeJitConstant("WITH_BIAS", "BIAS_TERM"));
    jit.AddConstant(MakeJitConstant("WITH_SUM", 0));

    jit.AddConstant(MakeJitConstant("MB_BLOCK", conf.mb_block));
    jit.AddConstant(MakeJitConstant("OC_BLOCK", conf.oc_block));
    jit.AddConstant(MakeJitConstant("IC_BLOCK", conf.ic_block));
    jit.AddConstant(MakeJitConstant("OH_BLOCK", conf.oh_block));
    jit.AddConstant(MakeJitConstant("OW_BLOCK", conf.ow_block));

    jit.AddConstant(MakeJitConstant("OW_LAST", rnd_dn(conf.ow, conf.ow_block)));
    jit.AddConstant(MakeJitConstant("OWB", CeilDiv(conf.ow, conf.ow_block)));
    jit.AddConstant(MakeJitConstant("OHB", CeilDiv(conf.oh, conf.oh_block)));
    jit.AddConstant(MakeJitConstant("IC_WO_PADDING", conf.ic_without_padding));
    jit.AddConstant(MakeJitConstant("OC_WO_PADDING", conf.oc_without_padding));
    jit.AddConstant(MakeJitConstant("OMB", conf.omb));
    jit.AddConstant(MakeJitConstant("OCB", conf.ocb));
    jit.AddConstant(MakeJitConstant("G", conf.ngroups));
    jit.AddConstant(MakeJitConstant("MB", conf.mb));
    jit.AddConstant(MakeJitConstant("OC", output.Feature().v / params.groups));
    jit.AddConstant(MakeJitConstant("OD", conf.od));
    jit.AddConstant(MakeJitConstant("OH", conf.oh));
    jit.AddConstant(MakeJitConstant("OW", conf.ow));
    jit.AddConstant(MakeJitConstant("IC", input.Feature().v / params.groups));
    jit.AddConstant(MakeJitConstant("ID", conf.id));
    jit.AddConstant(MakeJitConstant("IH", conf.ih));
    jit.AddConstant(MakeJitConstant("IW", conf.iw));
    jit.AddConstant(MakeJitConstant("KD", conf.kd));
    jit.AddConstant(MakeJitConstant("KH", conf.kh));
    jit.AddConstant(MakeJitConstant("KW", conf.kw));
    jit.AddConstant(MakeJitConstant("SD", conf.stride_d));
    jit.AddConstant(MakeJitConstant("SH", conf.stride_h));
    jit.AddConstant(MakeJitConstant("SW", conf.stride_w));
    jit.AddConstant(MakeJitConstant("PD", conf.f_pad));
    jit.AddConstant(MakeJitConstant("PH", conf.t_pad));
    jit.AddConstant(MakeJitConstant("PW", conf.l_pad));
    jit.AddConstant(MakeJitConstant("PD_R", conf.back_pad));
    jit.AddConstant(MakeJitConstant("PH_R", conf.b_pad));
    jit.AddConstant(MakeJitConstant("PW_R", conf.r_pad));

    jit.AddConstant(MakeJitConstant("IC_FULL", params.inputs[0].Feature().LogicalDimPadded()));
    jit.AddConstant(MakeJitConstant("ID_FULL", params.inputs[0].Z().LogicalDimPadded()));
    jit.AddConstant(MakeJitConstant("IH_FULL", params.inputs[0].Y().LogicalDimPadded()));
    jit.AddConstant(MakeJitConstant("IW_FULL", params.inputs[0].X().LogicalDimPadded()));

    jit.AddConstant(MakeJitConstant("OC_FULL", params.output.Feature().LogicalDimPadded()));
    jit.AddConstant(MakeJitConstant("OD_FULL", params.output.Z().LogicalDimPadded()));
    jit.AddConstant(MakeJitConstant("OH_FULL", params.output.Y().LogicalDimPadded()));
    jit.AddConstant(MakeJitConstant("OW_FULL", params.output.X().LogicalDimPadded()));

    return jit;
}

KernelsData ConvolutionKernel_b_fs_zyx_fsv16::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector
