// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>
#include <ie_layers.h>

#include <string>
#include <map>
#include <memory>
#include <algorithm>

namespace InferenceEngine {
namespace Extensions {
namespace Gpu {

class GPUExtensions : public IExtension {
public:
    GPUExtensions() {}
    typedef std::function<std::shared_ptr<ILayerImpl>(const std::shared_ptr<ngraph::Node>&, std::string)> FactoryType;

    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {
        static Version ExtensionDescription = {
            { 2, 0 },    // extension API version
            "2.0",
            "ie-gpu-ext"  // extension description message
        };

        versionInfo = &ExtensionDescription;
    }

    void Unload() noexcept override {}

    void Release() noexcept override {
        delete this;
    }

    template<typename ExtLayerType>
    void RegisterExtensionLayer(std::string layerType, std::string implType) {
        extensionLayers.insert({layerType, {implType,
        [](const std::shared_ptr<ngraph::Node>& op, std::string implType) -> std::shared_ptr<ExtLayerType> {
            return std::make_shared<ExtLayerType>(op, implType);
        }}});
    }

    template<typename ExtLayerType>
    void RegisterExtensionLayer(std::string layerType) {
        extensionLayers.insert({layerType, {"",
        [](const std::shared_ptr<ngraph::Node>& op, std::string implType = "") -> std::shared_ptr<ExtLayerType> {
            return std::make_shared<ExtLayerType>(op);
        }}});
    }

    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override;
    ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) override;

private:
    std::multimap<std::string, std::pair<std::string, FactoryType>> extensionLayers;
};

}  // namespace Gpu
}  // namespace Extensions
}  // namespace InferenceEngine
