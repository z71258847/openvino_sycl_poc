// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal GPU plugin-specific
 * OpenCL context and OpenCL shared memory tensors
 *
 * @file openvino/runtime/intel_gpu/ocl/ocl.hpp
 */
#pragma once

#include <memory>
#include <string>

#include "gpu/gpu_params.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/dpcpp/remote_properties.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/remote_tensor.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/backend/opencl.hpp>

namespace ov {

/**
 * @brief Namespace with Intel GPU OpenCL specific remote objects
 */
namespace dpcpp {

/**
 * @brief Shortcut for defining a handle parameter
 */
using gpu_handle_param = void*;

/**
 * @brief This class represents an abstraction for GPU plugin remote tensor
 * which can be shared with user-supplied USM device pointer.
 * The plugin object derived from this class can be obtained with DPCPPContext::create_tensor() call.
 * @note User can obtain USM pointer from this class.
 */
class USMTensor : public RemoteTensor {
public:
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param tensor a tensor to check
     */
    static void type_check(const Tensor& tensor) {
        std::vector<std::string> mem_types = {enum_to_string(SharedMemType::USM_HOST_BUFFER),
                                              enum_to_string(SharedMemType::USM_DEVICE_BUFFER),
                                              enum_to_string(SharedMemType::USM_USER_BUFFER)};

        RemoteTensor::type_check(tensor,
                                 {{mem_handle.name(), {}},
                                  {shared_mem_type.name(), mem_types}});
    }

    /**
     * @brief Returns the underlying USM pointer.
     * @return underlying USM pointer
     */
    void* get() {
        return static_cast<void*>(get_params().at(mem_handle.name()).as<void*>());
    }
};

/**
 * @brief This class represents an abstraction for GPU plugin remote context
 * which is shared with OpenCL context object.
 * The plugin object derived from this class can be obtained either with
 * CompiledModel::get_context() or Core::create_context() calls.
 */
class DPCPPContext : public RemoteContext {
public:
    // Needed to make create_tensor overloads from base class visible for user
    using RemoteContext::create_tensor;
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param remote_context A remote context to check
     */
    static void type_check(const RemoteContext& remote_context) {
        RemoteContext::type_check(remote_context,
                                  {{context_handle.name(), {}},
                                   {context_type.name(), {enum_to_string(ContextType::DPCPP)}}});
    }

    /**
     * @brief Constructs context object from user-supplied OpenCL context handle
     * @param core A reference to OpenVINO Runtime Core object
     * @param ctx A OpenCL context to be used to create shared remote context
     * @param ctx_device_id An ID of device to be used from ctx
     */
    DPCPPContext(Core& core, sycl::context ctx, int ctx_device_id = 0) {
        AnyMap context_params = {{context_type.name(), enum_to_string(ContextType::DPCPP)},
                                 {ocl_context.name(), static_cast<gpu_handle_param>(sycl::get_native<sycl::backend::opencl>(ctx))},
                                 {context_handle.name(), static_cast<gpu_handle_param>(&ctx)},
                                 {ocl_context_device_id.name(), ctx_device_id}};
        *this = core.create_context("GPU", context_params).as<DPCPPContext>();
    }


    /**
     * @brief Returns the underlying OpenCL context handle.
     * @return `cl_context`
     */
    sycl::context get() {
        ov::Any sycl_handle = get_params().at(context_handle.name());
        if (!sycl_handle.empty())
            return *static_cast<sycl::context*>(sycl_handle.as<gpu_handle_param>());

        auto ocl_handle = static_cast<cl_context>(get_params().at(ocl_context.name()).as<gpu_handle_param>());
        return sycl::make_context<sycl::backend::opencl>(ocl_handle);
    }

    /**
     * @brief Returns the underlying queue handle.
     * @return `sycl::queue`
     */
    sycl::queue get_queue() {
        ov::Any queue_handle = get_params().at(ocl_queue.name());
        if (!queue_handle.empty()) {
            auto ocl_handle = static_cast<cl_command_queue>(queue_handle.as<gpu_handle_param>());
            return sycl::make_queue<sycl::backend::opencl>(ocl_handle, get());
        } else {
            throw std::runtime_error("Cant get sycl queue from the context");
        }
    }

    /**
     * @brief OpenCL context handle conversion operator for the DPCPPContext object.
     * @return `sycl::context`
     */
    operator sycl::context() {
        return get();
    }

    /**
     * @brief This function is used to obtain remote tensor object from user-supplied USM pointer
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param usm_ptr A USM pointer that should be wrapped by a remote tensor
     * @return A remote tensor instance
     */
    USMTensor create_tensor(const element::Type type, const Shape& shape, void* usm_ptr) {
        AnyMap params = {{shared_mem_type.name(), enum_to_string(SharedMemType::USM_USER_BUFFER)},
                         {mem_handle.name(), static_cast<gpu_handle_param>(usm_ptr)}};
        return create_tensor(type, shape, params).as<USMTensor>();
    }

    /**
     * @brief This function is used to allocate USM tensor with host allocation type
     * @param type Tensor element type
     * @param shape Tensor shape
     * @return A remote tensor instance
     */
    USMTensor create_usm_host_tensor(const element::Type type, const Shape& shape) {
        AnyMap params = {{shared_mem_type.name(), enum_to_string(SharedMemType::USM_HOST_BUFFER)}};
        return create_tensor(type, shape, params).as<USMTensor>();
    }

    /**
     * @brief This function is used to allocate USM tensor with device allocation type
     * @param type Tensor element type
     * @param shape Tensor shape
     * @return A remote tensor instance
     */
    USMTensor create_usm_device_tensor(const element::Type type, const Shape& shape) {
        AnyMap params = {{shared_mem_type.name(), enum_to_string(SharedMemType::USM_DEVICE_BUFFER)}};
        return create_tensor(type, shape, params).as<USMTensor>();
    }
};

}  // namespace dpcpp
}  // namespace ov
