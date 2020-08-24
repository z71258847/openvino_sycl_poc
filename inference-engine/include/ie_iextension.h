// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for Inference Engine Extension Interface
 *
 * @file ie_iextension.h
 */
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_api.h"
#include "ie_common.h"
#include "ie_layouts.h"
#include "ie_blob.h"
#include "ie_version.hpp"
#include "details/ie_no_copy.hpp"

/**
 * @def INFERENCE_EXTENSION_API(TYPE)
 * @brief Defines Inference Engine Extension API method
 */

#if defined(_WIN32) && defined(IMPLEMENT_INFERENCE_EXTENSION_API)
#define INFERENCE_EXTENSION_API(TYPE) extern "C" __declspec(dllexport) TYPE
#else
#define INFERENCE_EXTENSION_API(TYPE) INFERENCE_ENGINE_API(TYPE)
#endif

namespace ngraph {

class OpSet;
class Node;

}  // namespace ngraph

namespace InferenceEngine {

/**
 * @struct DataConfig
 * @brief This structure describes data configuration
 */
struct DataConfig {
    /**
     * @brief Format of memory descriptor
     */
    TensorDesc desc;
    /**
     * @brief Index of in-place memory. If -1 memory cannot be in-place
     */
    int inPlace = -1;
    /**
     * @brief Flag for determination of the constant memory. If layer contains all constant memory we can calculate it
     * on the load stage.
     */
    bool constant = false;
};

/**
 * @struct LayerConfig
 * @brief This structure describes Layer configuration
 */
struct LayerConfig {
    /**
     * @brief Supported dynamic batch. If false, dynamic batch is not supported
     */
    bool dynBatchSupport = false;
    /**
     * @brief Vector of input data configs
     */
    std::vector<DataConfig> inConfs;
    /**
     * @brief Vector of output data configs
     */
    std::vector<DataConfig> outConfs;
};

/**
 * @interface ILayerImpl
 * @brief This class provides interface for extension implementations
 */
class INFERENCE_ENGINE_API_CLASS(ILayerImpl) {
public:
    /**
     * @brief A shared pointer to the ILayerImpl interface
     */
    using Ptr = std::shared_ptr<ILayerImpl>;

    /**
     * @brief Destructor
     */
    virtual ~ILayerImpl();
};

/**
 * @interface ILayerExecImpl
 * @brief This class provides interface for the implementation with the custom execution code
 */
class INFERENCE_ENGINE_API_CLASS(ILayerExecImpl) : public ILayerImpl {
public:
    /**
     * @brief A shared pointer to the ILayerExecImpl interface
     */
    using Ptr = std::shared_ptr<ILayerExecImpl>;

    /**
     * @brief Destructor
     */
    virtual ~ILayerExecImpl();

    /**
     * @brief Gets all supported configurations for the current layer
     *
     * @param conf Vector with supported configurations
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode getSupportedConfigurations(std::vector<LayerConfig>& conf, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Initializes the implementation
     *
     * @param config Selected supported configuration
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode init(LayerConfig& config, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Execute method
     *
     * @param inputs Vector of blobs with input memory
     * @param outputs Vector of blobs with output memory
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                               ResponseDesc* resp) noexcept = 0;
};

/**
 * @interface ILayerImplOCL
 * @brief This class provides interface for the OpenCL implementations for custom layers
 */
class INFERENCE_ENGINE_API_CLASS(ILayerImplOCL): public ILayerImpl {
public:
    typedef std::shared_ptr<ILayerImplOCL> Ptr;
    /**
     * @brief Base descriptor for kernel arguments
     */
    class ArgumentDesciptor {
    public:
        typedef std::shared_ptr<ArgumentDesciptor> Ptr;
        virtual ~ArgumentDesciptor() = default;
    };

    class InputTensor : public ArgumentDesciptor {
    public:
        typedef std::shared_ptr<InputTensor> Ptr;
        InputTensor(size_t index, std::vector<DataConfig> configs) : index(index), configs(configs) {}
        size_t index;
        std::vector<DataConfig> configs;
    };

    class OutputTensor : public ArgumentDesciptor {
    public:
        typedef std::shared_ptr<OutputTensor> Ptr;
        OutputTensor(size_t index, std::vector<DataConfig> configs) : index(index), configs(configs) {}
        size_t index;
        std::vector<DataConfig> configs;
    };

    template<typename T>
    class Scalar : public ArgumentDesciptor {
    public:
        Scalar(T value) : value(value) {}
        T value;
    };

    /**
     * @brief A structure that contains additional information for correct kernel build and run.
     */
    struct RuntimeInfo {
        std::vector<size_t> gws;                        //!< Global workgroup size
        std::vector<size_t> lws;                        //!< Local workgroup size
        std::string kernelName;                         //!< Entry point
        std::string buildOptions;                       //!< Build options for OCL kernel
        std::vector<ArgumentDesciptor::Ptr> arguments;  //!< Descriptors for kernel arguments
    };

    /**
     * @brief Helper structure to store user's macro definitions
     */
    struct JitConstant {
        std::string name;   //!< Macro name
        std::string value;  //!< Macro value
    };

    explicit ILayerImplOCL(const std::shared_ptr<ngraph::Node>& op, std::string device) : op(op), device(device) {}
    /**
     * @brief Returns runtime info for given operation.
     * @return runtime info for given operation.
     */
    virtual RuntimeInfo getRuntimeInfo() const = 0;

    /**
     * @brief Returns string with opencl code to execute given ngraph operation
     * @return string with kernel code
     */
    virtual std::string getKernelSource() const;

    /**
     * @brief Returns precompiled byte code with opencl kernel to execute given ngraph operation
     * Can be unimplemented in derived claases if cl source code is defined
     * @return byte array with the kernel object
     */
    virtual std::vector<char> getKernelBinary() const { return {}; }

protected:
    /**
     * @brief Returns string with a template code of opencl operation
     * Can be unimplemented in derived classes if precompiled binary is used
     * @return string with kernel template
     */
    virtual std::string getKernelTemplate() const = 0;
    /**
     * @brief Returns a vector of jit constant that should be additionaly defined in the kernel's source code
     * @return vector of jit constants
     */
    virtual std::vector<JitConstant> getJitConstants() const { return {}; };

    /**
     * @brief Shared pointer to ngraph Node for the custom layer
     */
    std::shared_ptr<ngraph::Node> op;

    /**
     * @brief Name of the device that impl was instantiated for.
     */
    std::string device;
};

/**
 * @brief This class is the main extension interface
 */
class INFERENCE_ENGINE_API_CLASS(IExtension) : public InferenceEngine::details::IRelease {
public:
    /**
     * @brief Returns operation sets
     * This method throws an exception if it was not implemented
     * @return map of opset name to opset
     */
    virtual std::map<std::string, ngraph::OpSet> getOpSets();

    /**
     * @brief Returns vector of implementation types
     * @param node shared pointer to nGraph op
     * @return vector of strings
     */
    virtual std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) {
        (void)node;
        return {};
    }

    /**
     * @brief Returns implementation for specific nGraph op
     * @param node shared pointer to nGraph op
     * @param implType implementation type
     * @return shared pointer to implementation
     */
    virtual ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) {
        (void)node;
        (void)implType;
        return nullptr;
    }

    /**
     * @brief Cleans resources up
     */
    virtual void Unload() noexcept = 0;

    /**
     * @brief Gets extension version information and stores in versionInfo
     * @param versionInfo Pointer to version info, will be set by plugin
     */
    virtual void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept = 0;
};

/**
 * @brief A shared pointer to a IExtension interface
 */
using IExtensionPtr = std::shared_ptr<IExtension>;

/**
 * @brief Creates the default instance of the extension
 *
 * @param ext Extension interface
 * @param resp Response description
 * @return Status code
 */
INFERENCE_EXTENSION_API(StatusCode) CreateExtension(IExtension*& ext, ResponseDesc* resp) noexcept;

}  // namespace InferenceEngine
