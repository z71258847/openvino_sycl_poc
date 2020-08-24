// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_iextension.h"

#include <ngraph/node.hpp>

namespace InferenceEngine {

std::string ILayerImplOCL::getKernelSource() const {
    auto kernelTemplate = getKernelTemplate();
    if (kernelTemplate.empty())
        return "";

    auto jit = getJitConstants();

    auto getDefineString = [](const JitConstant& jitConst) -> std::string {
        return "#define " + jitConst.name + " (" + jitConst.value + ")";
    };

    std::string jitStr = "";
    for (auto& jitConst : jit) {
        jitStr += getDefineString(jitConst) + "\n";
    }

    auto rt = getRuntimeInfo();
    const std::string layerTitle("\n// Layer " + op->get_friendly_name() + " using custom kernel " + rt.kernelName + "\n");
    const std::string defineTitle("// Custom Layer User Defines\n");
    return layerTitle + defineTitle + jitStr + "\n" + getKernelTemplate();
}

}  // namespace InferenceEngine
