// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_cpu.hpp"
#include "extension/executor.hpp"
#include "impls/convolution.hpp"


#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/reference/convolution.hpp"
#include "opset/convolution.hpp"

namespace {

template <ov::element::Type_t T>
bool evaluate(const ov::intel_gpu::op::Convolution* op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using ET = typename ov::element_type_traits<T>::value_type;
    const auto filter_data = inputs[1].data<ET>();
    auto out_data_ptr = outputs[0].data<ET>();
    const auto in_data_ptr = inputs[0].data<ET>();
    const auto& out_shape = outputs[0].get_shape();
    const auto& in_shape = inputs[0].get_shape();
    const auto& filter_shape = inputs[1].get_shape();
    ov::reference::convolution<ET>(in_data_ptr,
                                   filter_data,
                                   out_data_ptr,
                                   in_shape,
                                   filter_shape,
                                   out_shape,
                                   op->get_strides(),
                                   op->get_dilations(),
                                   op->get_pads_begin(),
                                   op->get_pads_end());
    return true;
}

bool evaluate_node(const ov::Node* node, ov::TensorVector& outputs, const ov::TensorVector& inputs) {
    auto element_type = node->get_input_element_type(0);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(dynamic_cast<const ov::intel_gpu::op::Convolution*>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}

}  // namespace

namespace ov {
namespace cpu {

class SomeConvolutionCPUExecutor : public OpExecutor {
public:
    explicit SomeConvolutionCPUExecutor(const SomeCustomParams* params) : m_params(params) { }

    Event::Ptr execute(Stream& stream, const MemoryArgs& args, const Events dep_events) override {
        std::cerr << "SomeConvolutionCPUExecutor::execute() " << (m_params != nullptr ? "with params" : "null params") << "\n";

        auto input = args.at(Argument::input(0));
        auto weights = args.at(Argument::weights());
        auto output = args.at(Argument::output(0));
        ov::TensorVector inputs {input->to_tensor(), weights->to_tensor()};
        ov::TensorVector outputs {output->to_tensor()};
        OPENVINO_ASSERT(m_params != nullptr);
        OPENVINO_ASSERT(m_params->m_node != nullptr);
        evaluate_node(m_params->m_node, outputs, inputs);


        return nullptr;
    }

private:
    const SomeCustomParams* m_params;
};


bool SomeConvolutionCPUImpl::supports(const ImplementationParameters* params) const {
    return true;
}

OpExecutor::Ptr SomeConvolutionCPUImpl::get_executor() const {
    auto typed_params = dynamic_cast<const SomeCustomParams*>(m_params);
    return std::make_shared<SomeConvolutionCPUExecutor>(typed_params);
}

}  // namespace cpu
}  // namespace ov
