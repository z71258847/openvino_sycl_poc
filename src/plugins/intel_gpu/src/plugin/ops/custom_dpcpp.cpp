// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/simple_math.hpp"

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/dpcpp/remote_properties.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl_wrapper.hpp"

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/node.hpp"

#include "intel_gpu/primitives/custom_dpcpp_primitive.hpp"
#include "intel_gpu/primitives/reorder.hpp"

using namespace InferenceEngine;

namespace ov {
namespace runtime {
namespace intel_gpu {


template<typename T>
static inline std::string vecToString(std::vector<T> vec) {
    if (vec.empty())
        return "";

    std::string res = std::to_string(vec[0]);
    for (size_t i = 1; i < vec.size(); i++) {
        res += "," + std::to_string(vec[i]);
    }
    return res;
}

template<>
inline std::string vecToString<std::string>(std::vector<std::string> vec) {
    if (vec.empty())
        return "";

    std::string res = vec[0];
    for (size_t i = 1; i < vec.size(); i++) {
        res += "," + vec[i];
    }
    return res;
}

void CreateCustomDPCPPOp(Program &p,
                         const std::shared_ptr<ngraph::Node>& op,
                         ov::DPCPPEvaluateExtension::Ptr evaluate_ext) {
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string genericLayerName = layer_type_name_ID(op);
    // TODO: Handle reordering input/output dims... Requires a way to describe input/output layout
    cldnn::custom_dpcpp_primitive::execute_function f = [op, genericLayerName, evaluate_ext](
            cldnn::stream& stream,
            const std::vector<cldnn::event::ptr>& dependent_events,
            const std::vector<cldnn::memory::ptr>& inputs,
            const std::vector<cldnn::memory::ptr>& outputs) {

        // TODO: we don't want to wait for the events, we just want the user to submit onto the queue
        for (auto& ev : dependent_events) {
            ev->wait();
        }

        ov::Core core;

        cl_context ctx;
        cl_command_queue queue = stream.get_cl_queue().get();
        if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr) != CL_SUCCESS)
            IE_THROW() << "Can't get context from given opencl queue";

        AnyMap context_params = {{ov::dpcpp::context_type.name(), enum_to_string(ov::dpcpp::ContextType::DPCPP)},
                                 {ov::dpcpp::ocl_context.name(),  static_cast<ov::dpcpp::gpu_handle_param>(ctx)},
                                 {ov::dpcpp::ocl_queue.name(),  static_cast<ov::dpcpp::gpu_handle_param>(queue)},
                                 {ov::dpcpp::context_handle.name(), nullptr},
                                 {ov::dpcpp::ocl_context_device_id.name(), 0}};

        auto context = core.create_context("GPU", context_params);

        // TODO: how do we have the user return an event?
        cldnn::event::ptr ev = stream.create_user_event(false);


        std::vector<ov::Tensor> input_tensors(inputs.size());

        auto memory_to_tensor = [&context](cldnn::memory::ptr mem) -> ov::Tensor {
            auto layout = mem->get_layout();
            auto et = element_type_from_data_type(layout.data_type);
            auto dims = layout.get_dims();
            ov::Shape shape(dims.begin(), dims.end());

            auto params = mem->get_internal_params();
            switch(mem->get_allocation_type()) {
                case cldnn::allocation_type::usm_device:
                case cldnn::allocation_type::usm_host: {
                    auto usm_ptr = static_cast<void*>(params.mem);
                    AnyMap params = {{ov::dpcpp::shared_mem_type.name(), enum_to_string(ov::dpcpp::SharedMemType::USM_USER_BUFFER)},
                                     {ov::dpcpp::mem_handle.name(), static_cast<ov::dpcpp::gpu_handle_param>(usm_ptr)}};
                    return context.create_tensor(et, shape, params);
                }
                default:
                    IE_THROW() << "clDNN: memory can't be shared...";
            }
        };

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto& input = inputs[i];
            input_tensors[i] = memory_to_tensor(input);
        }

        std::vector<ov::Tensor> output_tensors(outputs.size());
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            auto& output = outputs[i];
            output_tensors[i] = memory_to_tensor(output);
        }

        bool res = evaluate_ext->evaluate(op, output_tensors, input_tensors, context);

        ev->set();
        if (!res) {
            IE_THROW() << "Failed to execute custom evaluate for: " << genericLayerName;
        }
        return ev;
    };

    cldnn::layout outputLayout = cldnn::layout(DataTypeFromPrecision(op->get_output_element_type(0)),
                                               cldnn::format::any, // TODO: allow for user-defined input/output formats?
                                               tensor_from_dims(op->get_output_shape(0)));

    auto prim = cldnn::custom_dpcpp_primitive(genericLayerName, inputPrimitives, f, outputLayout, op->get_friendly_name());

    p.AddPrimitive(prim);
    p.AddPrimitiveToProfiler(op->get_friendly_name(), op);
    p.primitiveIDs[genericLayerName] = genericLayerName;
}

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
