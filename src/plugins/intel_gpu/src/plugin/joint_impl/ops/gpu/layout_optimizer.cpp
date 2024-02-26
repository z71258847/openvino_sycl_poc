// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layout_optimizer.hpp"

#include "intel_gpu/op/reorder.hpp"
#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/core/rt_info.hpp"

#include "openvino/core/type.hpp"
#include "openvino/op/binary_convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/deformable_convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/convolution_backprop_base.hpp"
#include "openvino/op/util/convolution_base.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/reduction_base.hpp"
#include "transformations/rt_info/is_shape_subgraph.hpp"
#include "transformations/utils/utils.hpp"
#include "joint_impl/node_extension.hpp"

#include "intel_gpu/runtime/debug_configuration.hpp"
#include <memory>


namespace ov {
namespace intel_gpu {
// namespace {

// std::set<ImplTypes> filter_impls(const DeviceInfo& device_info, const std::set<ImplTypes>& impls, const GPULayoutOptimizer::Attributes& attrs) {
//     std::set<ImplTypes> res{};
//     for (auto& impl : impls) {
//         if (impl == ImplTypes::onednn && (device_info.dev_type != device_type::discrete_gpu || !attrs.use_onednn))
//             continue;

//         res.insert(impl);
//     }
//     return res;
// }

// // void validate_impls(const program_node& node, const std::vector<ImplTypes>& requested_impls) {
// //     const auto& available_impls = node.get_available_impl_types();
// //     for (auto& requested_impl : requested_impls) {
// //         OPENVINO_ASSERT(available_impls.count(requested_impl) > 0, "[GPU] ", node.id(),  " node doesn't have ", requested_impl, " in a list of supported",
// //                                                                   " implementations. Available: ", to_str(available_impls));
// //     }
// // }

// void assign_expected_formats(const ov::op::util::ConvolutionFwdPropBase* conv_node, std::vector<Format>& in_fmts, std::vector<Format>& out_fmts) {
//     // TODO: implement me
//     // auto in_et = conv_node->get_input_element_type(0);
//     // bool i8_u8_input = in_et == element::u8 || in_et == element::i8;
//     in_fmts[0] = Format::b_fs_yx_fsv16;
//     out_fmts[0] = Format::b_fs_yx_fsv16;
// }

// void assign_expected_formats(const ov::op::util::ConvolutionBackPropBase* conv_node, std::vector<Format>& in_fmts, std::vector<Format>& out_fmts) {
//     // TODO: implement me
//     in_fmts[0] = Format::b_fs_yx_fsv16;
//     out_fmts[0] = Format::b_fs_yx_fsv16;
// }

// void assign_expected_formats(const ov::op::v0::FakeQuantize* fq_node, std::vector<Format>& in_fmts, std::vector<Format>& out_fmts) {
//     // TODO: implement me
//     in_fmts[0] = Format::b_fs_yx_fsv16;
//     out_fmts[0] = Format::b_fs_yx_fsv16;
// }

// bool is_rotating_except_batch(const ov::op::v1::Transpose* node) {
//     return false;
// }
// }  // namespace

GPULayoutOptimizer::GPULayoutOptimizer(const DeviceInfo& device_info, const ExecutionConfig& config, const Attributes& attrs)
    : m_device_info(device_info)
    , m_config(config)
    , m_attrs(attrs)
    , m_forcing_map(m_config.get_property(ov::intel_gpu::force_implementations)) { }

GPULayoutOptimizer::PreferredFormats GPULayoutOptimizer::get_preferred_formats(const std::shared_ptr<ov::Node>& node, ImplTypes impl_type) const {
    // init with any for all in/out ports
    std::vector<Format> preferred_in_fmts(node->get_input_size(), Format::any);
    std::vector<Format> preferred_out_fmts(node->get_output_size(), Format::any);

    // if (!m_forcing_map.empty() && m_forcing_map.count(node->get_friendly_name())) {
    //     auto forced_fmt = m_forcing_map.at(node->get_friendly_name()).output_format;
    //     preferred_in_fmts[0] = forced_fmt;
    //     preferred_out_fmts[0] = forced_fmt;

    //     return {preferred_in_fmts, preferred_out_fmts};
    // }

    // // if (node.is_type<input_layout>()) {
    // //     for (size_t i = 0; i < node.get_dependencies().size(); i++) {
    // //         preferred_in_fmts[i] = format::get_default_format(node.get_input_pshape(i).size());
    // //     }
    // //     for (size_t i = 0; i < node.get_outputs_count(); i++) {
    // //         preferred_out_fmts[i] = format::get_default_format(node.get_output_pshape(i).size());
    // //     }
    // //     return {preferred_in_fmts, preferred_out_fmts};
    // // }

    // if (impl_type == ImplTypes::onednn) {
    // //     return get_formats_for_onednn(node);
    // }

    // auto in0_rank = node->get_input_size() > 0 ? node->get_input_partial_shape(0).size() : 0;
    // auto out0_rank = node->get_output_size() > 0 ? node->get_output_partial_shape(0).size() : 0;

    // if (ov::op::util::is_parameter(node) || ov::op::util::is_constant(node)) {
    //     preferred_out_fmts[0] = Format::get_default_format(out0_rank);
    // } else if (ov::op::util::is_output(node)) {
    //     preferred_in_fmts[0] = Format::get_default_format(in0_rank);
    // } else if (ov::is_type<ov::op::util::DeformableConvolutionBase>(node)) {
    //     preferred_in_fmts[0] = Format::get_default_format(in0_rank);
    //     preferred_out_fmts[0] = Format::get_default_format(out0_rank);
    // } else if (ov::is_type<ov::op::util::ConvolutionFwdPropBase>(node)) {
    //     assign_expected_formats(ov::as_type_ptr<ov::op::util::ConvolutionFwdPropBase>(node).get(), preferred_in_fmts, preferred_out_fmts);
    // } else if (ov::is_type<ov::op::util::ConvolutionBackPropBase>(node)) {
    //     assign_expected_formats(ov::as_type_ptr<ov::op::util::ConvolutionBackPropBase>(node).get(), preferred_in_fmts, preferred_out_fmts);
    // } else if (ov::is_type<ov::op::v0::FakeQuantize>(node)) {
    //     assign_expected_formats(ov::as_type_ptr<ov::op::v0::FakeQuantize>(node).get(), preferred_in_fmts, preferred_out_fmts);
    // } else if (ov::is_type<ov::op::v1::Reshape>(node)) {
    //     preferred_in_fmts[0] = Format::get_default_format(in0_rank);
    //     preferred_out_fmts[0] = Format::get_default_format(out0_rank);
    // } else if (ov::is_type<ov::op::v6::MVN>(node)) { // need v0?
    //     if (in0_rank == 5 && node->get_input_element_type(0).is_real()) {
    //         preferred_in_fmts[0] = Format::get_default_format(in0_rank);
    //         preferred_out_fmts[0] = Format::get_default_format(out0_rank);
    //     }
    // } else if (ov::is_type<ov::op::v11::Interpolate>(node)) { // need v4 and v0?
    // //     // if the resample is in the last part of the network and there are no users using blocked format,
    // //     // it is better to reorder to bfyx before resample is done.
    // //     if (all_users_simple_format_until_output(node, node, 0, 10)) {
    // //         const auto& dim = format::dimension(node.get_output_layout().format);
    // //         expected = format::get_default_format(dim, false, false);
    // //     }
    // } else if (ov::is_type<ov::op::v1::Transpose>(node)) {
    //     auto in_node = node->get_input_node_ptr(0);
    //     if (ov::is_type<ov::op::util::ConvolutionFwdPropBase>(in_node)) {
    //         auto gpu_in_node = dynamic_cast<GPUOpExtension*>(in_node);
    //         auto fmt = gpu_in_node->get_preferred_output_fmt(0);
    //         if (is_rotating_except_batch(ov::as_type_ptr<ov::op::v1::Transpose>(node).get()) && fmt == Format::fs_b_yx_fsv32) {
    //             preferred_in_fmts[0] = preferred_out_fmts[0] = fmt;
    //         }
    //     }
    // } else if (ov::is_type<ov::op::util::ReductionBase>(node)) {
    //     if (node->get_input_partial_shape(0).is_dynamic()) {
    //         if (in0_rank > 4) {
    //             preferred_in_fmts[0] = Format::get_default_format(in0_rank);
    //             preferred_out_fmts[0] = Format::get_default_format(out0_rank);
    //         }
    //     }
    // } else if (ov::is_type<ov::op::v11::TopK>(node)) { // need v3 and v1?
    //     preferred_in_fmts[0] = Format::get_default_format(in0_rank);
    //     preferred_out_fmts[0] = Format::get_default_format(out0_rank);
    // } else if (ov::is_type<ov::op::v3::ShapeOf>(node) || ov::is_shape_subgraph(node)) { // need v3 and v1?
    //     preferred_in_fmts[0] = Format::any;
    //     preferred_out_fmts[0] = Format::get_default_format(out0_rank);
    // } else if (ov::is_type<ov::intel_gpu::op::Reorder>(node)) {
    //     auto reorder = std::dynamic_pointer_cast<ov::intel_gpu::op::Reorder>(node);
    //     auto out_port = node->get_input_source_output(0).get_index();
    //     preferred_in_fmts[0] = std::dynamic_pointer_cast<GPUOpExtension>(reorder->get_input_node_shared_ptr(0))->get_preferred_output_fmt(out_port);
    //     preferred_out_fmts[0] = reorder->get_output_format();
    // }
    // // } else if (node.is_type<reorder>() || node.is_type<input_layout>()) {
    // //     if (node.is_type<reorder>() && node.as<reorder>().get_primitive()->has_surface_input()) {
    // //         expected = format::nv12;
    // //     } else {
    // //         expected = node.get_output_layout().format;
    // //     }

    // //     preferred_in_fmts[0] = format::any;
    // //     preferred_out_fmts[0] = expected;

    // //     return {preferred_in_fmts, preferred_out_fmts};
    // // }

    return {preferred_in_fmts, preferred_out_fmts};
}

ImplTypes GPULayoutOptimizer::get_preferred_impl_type(const std::shared_ptr<ov::Node>& node) const {
    return ImplTypes::ocl;
}

ImplTypes GPULayoutOptimizer::get_preferred_impl_type(const std::shared_ptr<ov::Node>& node, Format preferred_format) const {
    return ImplTypes::ocl;
}

void GPULayoutOptimizer::select_preferred_formats(const std::shared_ptr<ov::Node>& node) const {
    // // if (ov::is_type<ov::op::v0::Constant>(node) ||
    // //     ov::is_type<ov::op::v0::Parameter>(node) ||
    // //     ov::is_type<ov::op::v0::Result>(node))
    // //     return;

    // auto gpu_op = std::dynamic_pointer_cast<GPUOpExtension>(node);
    // OPENVINO_ASSERT(gpu_op != nullptr, "[GPU] Operation ", node->get_friendly_name(), " is not wrapped with GPUOpExtension type");

    // // auto in_dt = node.get_dependencies().size() > 0 ? node.get_input_layout(0).data_type : data_types::f32;
    // // auto out_dt = node.get_outputs_count() > 0 ? node.get_output_layout(0).data_type : data_types::f32;
    // // auto impls = node.type()->query_available_impls(in_dt, out_dt, shape_types::static_shape);

    // std::set<ImplTypes> impls = {ImplTypes::ocl};

    // auto filtered_impls = filter_impls(m_device_info, impls, m_attrs);
    // GPU_DEBUG_TRACE << node->get_friendly_name() << " all impls: " << to_str(impls) << " filtered impls: " << to_str(filtered_impls) << std::endl;
    // gpu_op->set_available_impl_types(filtered_impls);

    // for (auto& impl : filtered_impls) {
    //     auto fmts = get_preferred_formats(node, impl);

    //     gpu_op->set_preferred_input_fmts(impl, fmts.first);
    //     gpu_op->set_preferred_output_fmts(impl, fmts.second);
    // }
    // auto preferred_impl = get_preferred_impl_type(node);
    // OPENVINO_ASSERT(preferred_impl != ImplTypes::any, "[GPU] Any impl type is unexpected here for node ", node->get_friendly_name());
    // gpu_op->set_preferred_impl_type(preferred_impl);
}

bool GPULayoutOptimizer::is_optimized_format(Format fmt) const {
    const std::vector<std::pair<Format::type, bool>> optimized_formats = {
        {Format::b_fs_yx_fsv16, true},
        {Format::b_fs_yx_fsv16, false},
        {Format::b_fs_zyx_fsv16, false},
        {Format::bs_fs_zyx_bsv16_fsv16, false},
        {Format::bs_fs_yx_bsv16_fsv16, false},
        {Format::fs_b_yx_fsv32, false}
    };

    return std::find_if(optimized_formats.begin(), optimized_formats.end(), [fmt](const std::pair<Format::type, bool>& f) {
        return f.first == fmt;
    }) != optimized_formats.end();
}
bool GPULayoutOptimizer::is_format_supported(const ov::Node* op, Format fmt) const {
    return true;
}

}  // namespace intel_gpu
}  // namespace ov
