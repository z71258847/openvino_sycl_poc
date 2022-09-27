// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_kernel_base.h"

namespace kernel_selector {
JitConstants SoftmaxKernelBase::GetJitConstants(const softmax_params& params,
                                                SoftmaxKernelBase::DispatchData dispatchData) const {
    JitConstants mem_consts = MakeBaseParamsJitConstants(params);

    mem_consts.AddConstants({MakeJitConstant("ALONG_" + toString(params.dim), "")});

    auto& input = params.inputs[0];
    auto x = toCodeString(input.X(), 5);
    auto y = toCodeString(input.Y(), 4);
    auto z = toCodeString(input.Z(), 3);
    auto w = toCodeString(input.W(), 2);
    auto f = toCodeString(input.Feature(), 1);
    auto b = toCodeString(input.Batch(), 0);

    auto multiply = [](std::vector<std::string> dims) -> std::string {
        std::string res = "(";
        for (size_t i = 0; i < dims.size(); i++) {
            auto& d = dims[i];
            res += d;
            if (i != dims.size() - 1)
                res += "*";
        }
        res += ")";
        return res;
    };


    std::string items_num = std::to_string(dispatchData.itemsNum);
    std::string dataSetsCount = std::to_string(dispatchData.dataSetsCount);
    std::string dataSetSize = std::to_string(dispatchData.dataSetSize);
    std::string leftovers = std::to_string(dispatchData.leftovers);
    std::string lws0 = std::to_string(dispatchData.lws[0]);
    if (input.is_dynamic()) {
        // flatten f and spatials
        items_num = multiply({f, w, z, y, x});
        lws0 = "(uint)get_local_size(0)";
        dataSetsCount = b;
        dataSetSize = items_num;
        // TODO: move to individual kernels as this works for bf_ref only
        leftovers = "("  + dataSetSize + ") % " + "(" + lws0 + ")";
    }

    mem_consts.AddConstants({
        MakeJitConstant("ITEMS_NUM", items_num),
        MakeJitConstant("LWS", lws0),
        MakeJitConstant("DATA_SETS_COUNT", dataSetsCount),
        MakeJitConstant("DATA_SET_SIZE", dataSetSize),
        MakeJitConstant("LEFTOVERS", leftovers),
    });

    return mem_consts;
}

SoftmaxKernelBase::DispatchData SoftmaxKernelBase::SetDefault(const softmax_params&) const {
    DispatchData dispatchData;

    dispatchData.gws[0] = 1;
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    dispatchData.leftovers = 0;
    dispatchData.itemsNum = 0;
    dispatchData.normIndex = 0;
    dispatchData.dataSetsCount = 0;
    dispatchData.dataSetSize = 0;

    return dispatchData;
}

bool SoftmaxKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::SOFT_MAX || o.GetType() != KernelType::SOFT_MAX) {
        return false;
    }

    return true;
}

KernelsData SoftmaxKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    const softmax_params& orgParams = static_cast<const softmax_params&>(params);
    KernelData kd = KernelData::Default<softmax_params>(params);

    kd.update_kernels_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const softmax_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.internalBufferSizes.clear();
        kd.internalBufferSizes.push_back(prim_params.inputs[0].PhysicalSizeInBytes());
        kd.internalBufferDataType = Datatype::F32;
    };


    auto dispatchData = SetDefault(orgParams);
    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    bool is_dynamic = orgParams.outputs[0].is_dynamic();

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     DEFAULT,
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     is_dynamic);

    if (is_dynamic) {
        auto& args = kernel.params.arguments;
        args.clear();
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        kd.internalBufferSizes.clear();
        kd.internalBufferSizes.push_back(orgParams.inputs[0].PhysicalSizeInBytes());
        kd.internalBufferDataType = Datatype::F32;
    }
    return {kd};
}

bool SoftmaxKernelBaseBF::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o)) {
        return false;
    }

    const softmax_params& params = static_cast<const softmax_params&>(p);
    const auto& input = params.inputs[0];

    if (!params.activations.empty()) {
        return false;
    }

    if (input.GetLayout() == DataLayout::bf || input.GetLayout() == DataLayout::fb) {
        return true;
    }

    switch (params.dim) {
        case SoftmaxDim::X:
            return input.Y().v == 1 && input.Z().v == 1 && input.Feature().v == 1;
        case SoftmaxDim::Y:
            return input.X().v == 1 && input.Z().v == 1 && input.Feature().v == 1;
        case SoftmaxDim::Z:
            return input.X().v == 1 && input.Y().v == 1 && input.Feature().v == 1;
        case SoftmaxDim::FEATURE:
            return input.X().v == 1 && input.Y().v == 1 && input.Z().v == 1;
        default:
            return false;
    }
}

SoftmaxKernelBase::DispatchData SoftmaxKernelBaseBF::SetDefault(const softmax_params& params) const {
    const auto& input = params.inputs[0];

    DispatchData dispatchData = Parent::SetDefault(params);

    auto flatten_input = input.FlattenFeatureAndSpatials();
    dispatchData.dataSetSize = flatten_input.Feature().v;
    dispatchData.dataSetsCount = input.Batch().v;

    return dispatchData;
}
}  // namespace kernel_selector
