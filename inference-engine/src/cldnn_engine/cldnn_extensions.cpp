// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_extensions.h"
#include "ngraph/node.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Gpu {

std::vector<std::string> GPUExtensions::getImplTypes(const std::shared_ptr<ngraph::Node>& node) {
    std::vector<std::string> implTypes = {};
    for (auto kv : extensionLayers) {
        if (kv.first == node->get_type_name()) {
            auto type = kv.second.first;

            if (std::find(implTypes.begin(), implTypes.end(), type) == implTypes.end()) {
                implTypes.push_back(type);
            }
        }
    }

    return implTypes;
}

ILayerImpl::Ptr GPUExtensions::getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) {
    std::vector<decltype(extensionLayers)::value_type> matches;
    std::copy_if(extensionLayers.begin(), extensionLayers.end(), std::back_inserter(matches),
    [&](const std::pair<std::string, std::pair<std::string, FactoryType>>& e) {
        if (e.first != node->get_type_name())
            return false;

        if (e.second.first != implType)
            return false;

        return true;
    });

    if (matches.empty())
        return nullptr;

    auto bestMatch = matches.front();

    return bestMatch.second.second(node, implType);
}

}  // namespace Gpu
}  // namespace Extensions
}  // namespace InferenceEngine
