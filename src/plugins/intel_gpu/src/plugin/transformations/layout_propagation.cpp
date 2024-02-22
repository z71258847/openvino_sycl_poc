// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layout_propagation.hpp"

#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/core/rt_info.hpp"

#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "transformations/utils/utils.hpp"
#include "joint_impl/node_extension.hpp"
#include "layout_optimizer.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {


using FormatsMap = std::map<ov::Node*, std::pair<std::vector<Format>, std::vector<Format>>>;

// namespace {

// bool is_special_type(const std::shared_ptr<ov::Node>& node) {
//     // return ov::is_type<ov::op::v0::Parameter>(node) ||
//     //        ov::is_type<ov::op::v0::Result>(node) ||
//     //        ov::is_type<ov::op::v0::Constant>(node);
//     return false;
// }

// void init_formats(const std::vector<std::shared_ptr<ov::Node>>& ops, FormatsMap& formats) {
//     for (auto& op : ops) {
//         if (auto n = std::dynamic_pointer_cast<GPUOpExtension>(op)) {
//             formats[op.get()] = std::make_pair(n->get_preferred_input_fmts(), n->get_preferred_output_fmts());
//         } else if (is_special_type(op)) {
//             std::vector<Format> in_fmts;
//             std::vector<Format> out_fmts;
//             for (size_t i = 0; i < op->get_input_size(); i++) {
//                 in_fmts.push_back(Format::get_default_format(op->get_input_partial_shape(i).size()));
//             }
//             for (size_t i = 0; i < op->get_output_size(); i++) {
//                 out_fmts.push_back(Format::get_default_format(op->get_output_partial_shape(i).size()));
//             }
//             formats[op.get()] = std::make_pair(in_fmts, out_fmts);
//         } else {
//             OPENVINO_THROW("[GPU] unexpected type");
//         }
//     }
// }

// Format select_best_fmt(const LayoutOptimizer& optimizer, const std::vector<Format>& fmts) {
//     Format best_fmt = Format::any;
//     for (size_t i = 0; i < fmts.size(); i++) {
//         if (fmts[i] != Format::any) {
//             if (best_fmt == Format::any || (!optimizer.is_optimized_format(best_fmt) && optimizer.is_optimized_format(fmts[i]))) {
//                 best_fmt = fmts[i];
//             }
//         }
//     }

//     return best_fmt;
// }

// void propagate_in_fmt_to_out(const LayoutOptimizer& optimizer, ov::Node* op, FormatsMap& formats) {
//     if (op->get_input_size() == 0)
//         return;

//     const auto& in_fmts = formats[op].first;
//     auto& out_fmts = formats[op].second;
//     // Propagate from input to output if input is not any and output is any
//     for (size_t out_port = 0; out_port < op->get_output_size(); out_port++) {
//         if (out_fmts[out_port] != Format::any)
//             continue;

//         // Try to pick best Format for output based on inputs
//         auto target_out_fmt = select_best_fmt(optimizer, in_fmts);
//         std::cerr << op->get_friendly_name() << " best out fmt: " << target_out_fmt << std::endl;

//         // Skip if haven't found a good one for now
//         if (target_out_fmt == Format::any)
//             continue;

//         out_fmts[out_port] = Format::adjust_to_rank(target_out_fmt, op->get_output_partial_shape(out_port).size());

//         std::cerr << op->get_friendly_name() << " assign output " << out_port << " Format "
//                   << out_fmts[out_port] << " propagated (forward) from input ports" << std::endl;
//     }
// }

// void propagate_out_fmt_to_in(const LayoutOptimizer& optimizer, ov::Node* op, FormatsMap& formats) {
//     auto& in_fmts = formats[op].first;
//     const auto& out_fmts = formats[op].second;
//     // Propagate from output to input if input is not any and output is any
//     for (size_t in_port = 0; in_port < op->get_input_size(); in_port++) {
//         if (in_fmts[in_port] != Format::any || ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(in_port)))
//             continue;

//         // Try to pick best Format for input based on utputs
//         auto target_in_fmt = select_best_fmt(optimizer, out_fmts);
//         std::cerr << op->get_friendly_name() << " best in fmt: " << target_in_fmt << std::endl;

//         // Skip if haven't found a good one for now
//         if (target_in_fmt == Format::any)
//             continue;

//         in_fmts[in_port] = Format::adjust_to_rank(target_in_fmt, op->get_input_partial_shape(in_port).size());

//         std::cerr << op->get_friendly_name() << " assign input " << in_port << " Format "
//                   << in_fmts[in_port] << " propagated (backward) from output ports" << std::endl;
//     }
// }

// void propagate_formats_forward(const LayoutOptimizer& optimizer, ov::Node* op, FormatsMap& formats);
// void propagate_formats_backward(const LayoutOptimizer& optimizer, ov::Node* op, FormatsMap& formats);

// void propagate_fmt_to_user(const LayoutOptimizer& optimizer, ov::Node* op, FormatsMap& formats) {
//     auto& out_fmts = formats[op].second;

//     for (size_t out_port = 0; out_port < out_fmts.size(); out_port++) {
//         if (out_fmts[out_port] == Format::any)
//             continue;

//         auto consumers = op->get_output_target_inputs(out_port);
//         for (auto& consumer : consumers) {
//             auto user = consumer.get_node();
//             auto in_idx = consumer.get_index();
//             auto& user_in_fmts = formats[user].first;
//             // Skip if there is some preferred Format assigned for this port
//             if (user_in_fmts[in_idx] != Format::any)
//                 continue;

//             if (!optimizer.is_format_supported(user, out_fmts[out_port]))
//                 continue;


//             user_in_fmts[in_idx] = out_fmts[out_port];

//             // If no preferences, then propagate Format of input
//             std::cerr << user->get_friendly_name() << " assign input " << in_idx << " Format "
//                       << out_fmts[out_port] << " propagated (forward) from " << op->get_friendly_name() << std::endl;
//             propagate_formats_forward(optimizer, user, formats);
//         }
//     }
// }

// void propagate_fmt_to_dependency(const LayoutOptimizer& optimizer, ov::Node* node, FormatsMap& formats) {
//     auto& in_fmts = formats[node].first;

//     for (size_t in_port = 0; in_port < in_fmts.size(); in_port++) {
//         if (in_fmts[in_port] == Format::any)
//             continue;

//         auto producer = node->get_input_source_output(in_port);
//         auto dep_node = producer.get_node_shared_ptr();

//         if (is_special_type(dep_node))
//             continue;

//         // Skip if there is some preferred Format assigned for this port
//         auto& dependency_out_fmts = formats[dep_node.get()].second;
//         if (dependency_out_fmts[producer.get_index()] != Format::any)
//             continue;

//         if (!optimizer.is_format_supported(dep_node.get(), in_fmts[in_port]))
//             continue;

//         // If no preferences, then propagate Format of input
//         dependency_out_fmts[producer.get_index()] = in_fmts[in_port];
//         std::cerr << dep_node->get_friendly_name() << " assign output Format "
//                   << in_fmts[in_port] << " propagated (backward) from " << node->get_friendly_name() << std::endl;
//         propagate_formats_backward(optimizer, dep_node.get(), formats);
//     }
// }

// void propagate_formats_forward(const LayoutOptimizer& optimizer, ov::Node* op, FormatsMap& formats) {
//     propagate_in_fmt_to_out(optimizer, op, formats);
//     propagate_fmt_to_user(optimizer, op, formats);
// }

// void propagate_formats_backward(const LayoutOptimizer& optimizer, ov::Node* op, FormatsMap& formats) {
//     propagate_out_fmt_to_in(optimizer, op, formats);
//     propagate_fmt_to_dependency(optimizer, op, formats);
// }

// void propagate_formats(const std::vector<std::shared_ptr<ov::Node>>& ops, const LayoutOptimizer& optimizer, FormatsMap& formats) {
//     for (auto& op : ops) {
//         if (is_special_type(op))
//             continue;
//         propagate_formats_forward(optimizer, op.get(), formats);
//         propagate_formats_backward(optimizer, op.get(), formats);
//     }
// }

// void optimize_formats(const std::vector<std::shared_ptr<ov::Node>>& ops, const LayoutOptimizer& optimizer, FormatsMap& formats) {
// //     // auto optimize_conv_permute = [&](convolution_node& node) {
// //     //     if (node.get_preferred_impl_type() != impl_types::onednn)
// //     //         return;

// //     //     // In conv-permute pattern, sets the output Format of conv to byxf so that permute can be optimized.
// //     //     // ex) oneDNN convolution -> (byxf) -> permute -> (bfyx) -> output
// //     //     //     output layout of convolution: byxf [b:1, f:128, y:2, x:2]
// //     //     //     output layout of permute:     bfyx [b:1, f:2, y:2, x:128]
// //     //     // In this case, it can be handled by changing only the shape of permute without the kernel execution.
// //     //     if (node.get_output_layout().get_rank() == 4
// //     //         && node.get_users().size() == 1 && node.get_users().front()->is_type<permute>()) {
// //     //         auto& pnode = node.get_users().front()->as<permute>();
// //     //         auto can_optimize_permute = pnode.get_output_layout().data_type == node.get_output_layout().data_type
// //     //             && !pnode.has_fused_primitives()
// //     //             && !pnode.is_output() && pnode.get_input_layout(0).is_static()
// //     //             && pnode.is_rotating_except_batch();
// //     //         if (can_optimize_permute) {
// //     //             formats[&node].second[0] = cldnn::Format::byxf;
// //     //             formats[&pnode].first[0] = cldnn::Format::byxf;
// //     //             formats[&pnode].second[0] = cldnn::Format::bfyx;
// //     //             std::cerr << "Optimize permute node: " << pnode.id() << " and update layout of previous conv " << node.id() << " to byxf\n";
// //     //             pnode.can_be_optimized(true);
// //     //         }
// //     //     }
// //     // };

// //     for (auto& node : p.get_processing_order()) {
// //         // program_helpers::do_for_types<convolution>(*node, optimize_conv_permute);

// //         if (!node->is_type<convolution>())
// //             continue;
// //         if (node->get_preferred_impl_type() != impl_types::onednn)
// //             continue;

// //         // In conv-permute pattern, sets the output Format of conv to byxf so that permute can be optimized.
// //         // ex) oneDNN convolution -> (byxf) -> permute -> (bfyx) -> output
// //         //     output layout of convolution: byxf [b:1, f:128, y:2, x:2]
// //         //     output layout of permute:     bfyx [b:1, f:2, y:2, x:128]
// //         // In this case, it can be handled by changing only the shape of permute without the kernel execution.
// //         if (node->get_output_layout().get_rank() == 4
// //             && node->get_users().size() == 1 && node->get_users().front()->is_type<permute>()) {
// //             auto& pnode = node->get_users().front()->as<permute>();
// //             auto can_optimize_permute = pnode.get_output_layout().data_type == node->get_output_layout().data_type
// //                 && !pnode.has_fused_primitives()
// //                 && !pnode.is_output() && pnode.get_input_layout(0).is_static()
// //                 && pnode.is_rotating_except_batch();
// //             if (can_optimize_permute) {
// //                 auto& conv_out_fmts = formats[node].second;
// //                 auto& permute_in_fmts = formats[node->get_users().front()].first;
// //                 auto& permute_out_fmts = formats[node->get_users().front()].second;
// //                 conv_out_fmts[0] = cldnn::Format::byxf;
// //                 permute_in_fmts[0] = cldnn::Format::byxf;
// //                 permute_out_fmts[0] = cldnn::Format::bfyx;
// //                 std::cerr << "Optimize permute node: " << pnode.id() << " and update layout of previous conv " << node->id() << " to byxf\n";
// //                 pnode.can_be_optimized(true);
// //             }
// //         }

// //     }
// }

// void finalize_input_formats(const ov::Node* op, std::vector<Format>& current_input_fmts, const std::vector<Format>& propagated_input_fmts) {
//     for (size_t i = 0; i < op->get_input_size(); i++) {
//         auto propagated_in_fmt = propagated_input_fmts[i];
//         if (propagated_in_fmt != Format::any) {
//             current_input_fmts[i] = propagated_in_fmt;
//         } else {
//             current_input_fmts[i] = Format::get_default_format(op->get_input_partial_shape(i).size());
//         }
//     }
// }

// void finalize_output_formats(const ov::Node* op, std::vector<Format>& current_output_fmts, const std::vector<Format>& propagated_output_fmts) {
//     for (size_t i = 0; i < op->get_output_size(); i++) {
//         auto propagated_out_fmt = propagated_output_fmts[i];
//         if (propagated_out_fmt != Format::any) {
//             current_output_fmts[i] = propagated_out_fmt;
//         } else {
//             current_output_fmts[i] = Format::get_default_format(op->get_output_partial_shape(i).size());
//         }
//     }
// }

// void finalize_formats(const std::vector<std::shared_ptr<ov::Node>>& ops, const LayoutOptimizer& optimizer, FormatsMap& formats) {
//     for (auto& op : ops) {
//         if (is_special_type(op))
//             continue;
//         auto gpu_node = std::dynamic_pointer_cast<GPUOpExtension>(op);
//         auto impls = gpu_node->get_available_impl_types();
//         if (impls.empty()) {
//             optimizer.select_preferred_formats(op);
//         }
//         impls = gpu_node->get_available_impl_types();
//         for (const auto& impl : impls) {
//             finalize_input_formats(op.get(), gpu_node->get_preferred_input_fmts(impl), formats[op.get()].first);
//             finalize_output_formats(op.get(), gpu_node->get_preferred_output_fmts(impl), formats[op.get()].second);
//         }

//         auto new_impl_type = optimizer.get_preferred_impl_type(op);
//         if (new_impl_type != gpu_node->get_preferred_impl_type()) {
//             GPU_DEBUG_TRACE << op->get_friendly_name() << " change impls from " << gpu_node->get_preferred_impl_type()
            //  << " to " << new_impl_type << std::endl;
//         }
//         gpu_node->set_preferred_impl_type(new_impl_type);
//     }
// }

// void print_fmts(const std::vector<std::shared_ptr<ov::Node>>& ops, const FormatsMap& formats) {
//     std::cerr << "-------------------------------------------------------------\n";
//     for (const auto& op : ops) {
//         // const auto& name = op->get_friendly_name();
//         if (!is_special_type(op)) {
//             auto gpu_node = std::dynamic_pointer_cast<GPUOpExtension>(op);
//             // std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!! " << name << " available impl: " << to_str(gpu_node->get_available_impl_types()) << std::endl;
//             // std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!! " << name << " impl: " << gpu_node->get_preferred_impl_type() << std::endl;
//             // std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!! " << name << " in: " << to_str(gpu_node->get_preferred_input_fmts()) << std::endl;
//             // std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!! " << name << " out: " << to_str(gpu_node->get_preferred_output_fmts())<< std::endl;
//         }
//         // std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!! " << name << " cur_in: " << to_str(formats.at(op.get()).first) << std::endl;
//         // std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!! " << name << " cur_out: " << to_str(formats.at(op.get()).second)<< std::endl;
//     }
// }
// }  // namespace

bool LayoutPropagation::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // const auto& ops = model->get_ordered_ops();

    // FormatsMap formats;
    // init_formats(ops, formats);
    // propagate_formats(ops, m_optimizer, formats);
    // print_fmts(ops, formats);
    // optimize_formats(ops, m_optimizer, formats);
    // print_fmts(ops, formats);
    // finalize_formats(ops, m_optimizer, formats);
    // print_fmts(ops, formats);

    // for (const auto& op : ops) {
    //     if (is_special_type(op))
    //         continue;

    //     auto n = std::dynamic_pointer_cast<GPUOpExtension>(op);
    //     for (const auto& fmt : n->get_preferred_input_fmts()) {
    //         OPENVINO_ASSERT(fmt != Format::any, "[GPU] ", op->get_friendly_name(), " Format::any is assigned to input for ", n->get_preferred_impl_type());
    //     }
    //     for (const auto& fmt : n->get_preferred_output_fmts()) {
    //         OPENVINO_ASSERT(fmt != Format::any, "[GPU] ",  op->get_friendly_name(), " Format::any is assigned to output for ", n->get_preferred_impl_type());
    //     }
    // }

    return false;
}

}  // namespace intel_gpu
}  // namespace ov
