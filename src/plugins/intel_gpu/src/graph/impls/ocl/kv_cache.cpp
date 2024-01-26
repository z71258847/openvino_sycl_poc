// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/ocl/kernel_selector_helper.h"
#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "kernel_selector_common.h"
#include "milti_state_primitive.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "kv_cache_inst.h"
#include "concatenation/concatenation_kernel_selector.h"
#include "concatenation/concatenation_kernel_base.h"
#include "gather/gather_kernel_selector.h"
#include "gather/gather_kernel_ref.h"
#include "openvino/core/partial_shape.hpp"

namespace cldnn {
namespace ocl {

namespace {
template<typename T>
T convert_axis(int64_t axis, size_t rank) {
    auto cldnn_axis = axis >= 0 ? axis : axis + static_cast<int64_t>(rank);
    if (cldnn_axis >= static_cast<int64_t>(rank))
        OPENVINO_THROW("kv_cache axis exceeds number of dimensions");

    // Difference in dimension ordering between IE and GPU plugin,
    // reverse spatial dimensions after batch and feature.
    if (cldnn_axis >= 2) {
        auto spatial_axis = cldnn_axis - 2;
        // Default and minimum number of dimensions is 4
        auto spatial_size = std::max<size_t>(rank, 4) - 2;
        cldnn_axis = spatial_size - spatial_axis - 1 + 2;
    }

    switch (cldnn_axis) {
        case 0: return T::BATCH;
        case 1: return T::FEATURE;
        case 2: return T::X;
        case 3: return T::Y;
        case 4: return T::Z;
        case 5: return T::W;
        default: OPENVINO_THROW("Unsupported kv_cache axis: ", axis);
    }

    return T::FEATURE;  // shouldn't get here
}

}  // namespace

struct kv_cache_impl : multi_stage_primitive<kv_cache> {
    using parent = multi_stage_primitive<kv_cache>;
    using parent::parent;

    using concat_kernel_selector_t = kernel_selector::concatenation_kernel_selector;
    using concat_kernel_params_t = std::pair<kernel_selector::concatenation_params, kernel_selector::concatenation_optional_params>;

    using gather_kernel_selector_t = kernel_selector::gather_kernel_selector;
    using gather_kernel_params_t = std::pair<kernel_selector::gather_params, kernel_selector::gather_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::kv_cache_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<kv_cache_impl>(*this);
    }

    std::vector<int32_t> beam_table_host;

    void set_arguments_impl(kv_cache_inst& instance) override { return; }

    kernel_arguments_data get_arguments(const kv_cache_inst& instance, size_t stage) const override {
        kernel_arguments_data args;
        args.shape_info = instance.shape_info_memory_ptr();
        const auto& desc = instance.get_typed_desc<kv_cache>();
        if (desc->indirect) {
            args.inputs = { instance.past_state[0], instance.input_memory_ptr(1) };
            args.outputs = {  instance.present_state[0] };
        } else {
            switch (stage) {
                case 0: {
                    args.inputs = { instance.past_state[1], instance.input_memory_ptr(2) };
                    args.outputs = {  instance.present_state[1] };
                    break;
                }
                case 1: {
                    args.inputs = { instance.past_state[0], instance.input_memory_ptr(1) };
                    args.outputs = {  instance.present_state[0] };
                    break;
                }
                default: OPENVINO_THROW("[GPU] Wrong stage for kv cache impl");
            }
        }

        return args;
    }

    std::vector<event::ptr> execute_stage(const std::vector<event::ptr>& events, kv_cache_inst& instance, std::vector<event::ptr>& all_events, size_t stage) {
        stream& stream = instance.get_network().get_stream();
        std::vector<event::ptr> tmp_events(events);
        size_t kernel_offset = 0;
        for (size_t s = 0; s < stage; s++) {
            kernel_offset += _kernels_data[s].kernels.size();
        }
        for (size_t kd_idx = 0; kd_idx < _kernels_data[stage].kernels.size(); ++kd_idx) {
            if (_kernels_data[stage].kernels[kd_idx].skip_execution)
                continue;

            size_t idx_final = kernel_offset + kd_idx;
            // If any user of the prim's users is CPU implementation or network's output, set prim as a output event (event won't be nullptr)
            bool needs_completion_event = instance.needs_completion_event();

            auto& params = _kernels_data[stage].kernels[kd_idx].params;
            auto args = get_arguments(instance, stage);
            args.scalars = &params.scalars;

            for (const auto& m : instance.get_intermediates_memories()) {
                args.intermediates.push_back(m);
            }

            stream.set_arguments(*_kernels[idx_final], _kernels_data[stage].kernels[kd_idx].params, args);

            const auto& gws = params.workGroups.global;
            const auto& lws = params.workGroups.local;

            GPU_DEBUG_TRACE_DETAIL << "Enqueue stage " << stage << " kernel " << idx_final << ": gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] "
                                   << "lws=[" << lws[0] << ", " << lws[1] << ", " << lws[2] << "]"
                                   << (needs_completion_event ? " has_completion_event=true" : "") << std::endl;

            auto ev = stream.enqueue_kernel(*_kernels[idx_final], params, args, tmp_events, needs_completion_event);
            stream.enqueue_barrier();
            if (_kernels_data[stage].needs_sub_kernels_sync) {
                tmp_events = {ev};
            }
            all_events.push_back(ev);
        }

        return all_events;
    }

    void update_beam_table(kv_cache_inst& instance, ov::intel_gpu::GPUVariableState& variable) {
        const auto& desc = instance.get_typed_desc<kv_cache>();
        if (!desc->indirect)
            return;

        auto& engine = instance.get_network().get_engine();
        stream& stream = instance.get_network().get_stream();

        ov::intel_gpu::VariableState::Ptr beam_table_state = nullptr;
        auto bt_layout = instance.get_impl_params()->output_layouts[1];
        OPENVINO_ASSERT(instance.present_state.size() == 2);
        OPENVINO_ASSERT(instance.past_state.size() == 2);
        beam_table_state = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCache&>(variable).get_beam_table_state();

        instance.past_state[1] = beam_table_state->get_memory();
        instance.present_state[1] = engine.allocate_memory(instance.get_impl_params()->output_layouts[1], false);
        instance.set_output_memory(instance.present_state[1], false, 1);

        if (!variable.is_set()) {
            std::cerr << "first infer beam table out shape: " << bt_layout.to_short_string() << std::endl;
            // initialize beam table with
            // [0, 0, 0, ... ]
            // [1, 1, 1, ... ]
            // ...
            // [beam_size, beam_size, beam_size, ... ]
            beam_table_host.resize(bt_layout.count());
            for (size_t i = 0; i < beam_table_host.size(); i++) {
                beam_table_host[i] = i / bt_layout.get_partial_shape()[1].get_length();
            }
        } else {
            std::cerr << "second+ infer beam table out shape: " << bt_layout.to_short_string() << " mem shape: "
                << instance.present_state[1]->get_layout().to_short_string() << std::endl;
            std::vector<int32_t> beam_table_tmp(bt_layout.count());
            mem_lock<int32_t, mem_lock_type::read> beam_idx(instance.dep_memory_ptr(2), stream);
            auto bt_shape = bt_layout.get_shape();
            for (size_t b = 0; b < bt_shape[0]; b++) {
                for (size_t s = 0; s < bt_shape[1] - 1; s++) {
                    beam_table_tmp[b*bt_shape[1] + s] = beam_table_host[beam_idx[b]*(bt_shape[1] - 1) + s];
                }
                beam_table_tmp[(b+1)*bt_shape[1] - 1] = b;
            }

            beam_table_host = beam_table_tmp;
        }

        instance.present_state[1]->copy_from(stream, beam_table_host.data(), false);
        beam_table_state->set_memory(instance.present_state[1], instance.get_impl_params()->output_layouts[1]);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, kv_cache_inst& instance) override {
        const auto& desc = instance.get_typed_desc<kv_cache>();
        stream& stream = instance.get_network().get_stream();
        auto& variable = instance.get_network().get_variable(desc->variable_info.variable_id);
        auto& engine = instance.get_network().get_engine();
        const auto impl_params = instance.get_impl_params();
        const bool can_be_optimized = impl_params->_can_be_optimized;

        std::vector<event::ptr> all_events;
        instance.past_state[0] = variable.get_memory();
        instance.present_state[0] = instance.output_memory_ptr(0);

        if (instance.past_state[0] && engine.is_the_same_buffer(*instance.present_state[0], *instance.past_state[0]) && !can_be_optimized) {
            const auto alloc_type = engine.get_preferred_memory_allocation_type(false);
            const bool reset = false;
            const auto present_layout = impl_params->output_layouts[0];
            instance.present_state[0] = engine.allocate_memory(present_layout, alloc_type, reset);
            instance.set_output_memory(instance.present_state[0], false, 0);
        }

        if (desc->indirect) {
            const size_t kv_concat_stage = 0;
            update_beam_table(instance, variable); // handle beam table
            execute_stage(events, instance, all_events, kv_concat_stage); // and execute KV cache concat
        } else {
            const size_t kv_concat_stage = 1;
            // do explicit gather
            //
            execute_stage(events, instance, all_events, kv_concat_stage); // and execute KV cache concat
        }

        variable.set_memory(instance.present_state[0], impl_params->output_layouts[0]);
        variable.set();

        stream.enqueue_barrier();
        return aggregate_events(all_events, stream, all_events.size() > 1);
    }

    static layout get_kv_layout(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<kv_cache>();
        auto init_shape = primitive->variable_info.data_shape;
        padding pad;
        pad.set_dynamic_pad(impl_param.output_layouts[0].data_padding.get_dynamic_pad_dims());
        layout initial_layout = {init_shape, primitive->variable_info.data_type, format::get_default_format(init_shape.size()), pad};
        return impl_param.state_layout.value_or(initial_layout);
    }

    static layout get_beam_table_layout(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<kv_cache>();
        auto kv_layout = get_kv_layout(impl_param);

        // expected to be normalized already on primitive creation
        auto concat_axis = primitive->concat_axis;
        auto gather_axis = primitive->concat_axis;

        auto kv_shape = kv_layout.get_partial_shape();
        auto beam_table_shape = ov::PartialShape{kv_shape[gather_axis], kv_shape[concat_axis]};
        padding pad;
        if (impl_param.output_layouts[0].data_padding.get_dynamic_pad_dims() != tensor(0)) {
            auto dyn_pad = tensor(0);
            dyn_pad.feature[0] = 1;
            pad.set_dynamic_pad(dyn_pad);
        }

        return layout{beam_table_shape, impl_param.output_layouts[1].data_type, format::get_default_format(beam_table_shape.size()), pad};
    }

    static concat_kernel_params_t get_kv_concat_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<kv_cache>();
        auto params = get_default_params<kernel_selector::concatenation_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::concatenation_optional_params>(impl_param.get_program());
        auto axis = primitive->concat_axis;

        const auto inputs_count = 2;
        params.inputs.resize(inputs_count);

        params.inputs[0] = convert_data_tensor(get_kv_layout(impl_param));
        params.inputs[1] = convert_data_tensor(impl_param.input_layouts[1]);

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset; // [kv_past, kv_initializer, kv_new_token, beam_table_past, beam_idx]
        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset; // [kv_present, beam_table_present]
        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(0)},
            {1, in_offsets_map.at(2)},
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, out_offsets_map.at(0)},
        };

        params.axis = convert_axis<kernel_selector::concat_axis>(axis, impl_param.get_output_layout().get_rank());
        optional_params.kernelPerInput = true;
        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return {params, optional_params};
    }

    static gather_kernel_params_t get_gather_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<kv_cache>();
        auto params = get_default_params<kernel_selector::gather_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::gather_optional_params>(impl_param.get_program());

        const auto inputs_count = 2;
        params.inputs.resize(inputs_count);
        params.inputs[0] = convert_data_tensor(get_beam_table_layout(impl_param)); // past beam table (state)
        params.inputs[1] = convert_data_tensor(impl_param.input_layouts[2]); // beam idx
        params.outputs[0] = convert_data_tensor(impl_param.output_layouts[1]); // present beam table

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset; // [kv_past, kv_initializer, kv_new_token, beam_table_past, beam_idx]
        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset; // [kv_present, beam_table_present]
        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(3)},
            {1, in_offsets_map.at(4)},
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, out_offsets_map.at(1)},
        };

        params.axis = kernel_selector::gather_axis::BATCH;
        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return {params, optional_params};
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<kv_cache>& arg, const kernel_impl_params& impl_param) {
        std::vector<kernel_selector::kernel_data> kernels_data;
        // Stage 0 for direct case
        if (!impl_param.typed_desc<kv_cache>()->indirect) {
            auto gather_kernel_params = get_gather_params(impl_param, impl_param.is_dynamic());
            auto& kernel_selector = gather_kernel_selector_t::Instance();
            kernels_data.push_back(kernel_selector.get_best_kernel(gather_kernel_params.first, gather_kernel_params.second));
        }

        // Stage 1 for both direct and indirect cases
        {
            auto kv_cache_kernel_params = get_kv_concat_params(impl_param, impl_param.is_dynamic());
            auto& kernel_selector = concat_kernel_selector_t::Instance();
            kernels_data.push_back(kernel_selector.get_best_kernel(kv_cache_kernel_params.first, kv_cache_kernel_params.second));
        }
        return make_unique<kv_cache_impl>(kernels_data);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        const bool indirect = impl_param.typed_desc<kv_cache>()->indirect;
        const size_t beam_table_stage = 0;
        const size_t kv_stage = indirect ? 0 : 1;
        if (!indirect) {
            auto gather_kernel_params = get_gather_params(impl_param, impl_param.is_dynamic());
            (_kernels_data[beam_table_stage].update_dispatch_data_func)(gather_kernel_params.first, _kernels_data[beam_table_stage]);
        }
        {
            auto kv_cache_kernel_params = get_kv_concat_params(impl_param, impl_param.is_dynamic());
            (_kernels_data[kv_stage].update_dispatch_data_func)(kv_cache_kernel_params.first, _kernels_data[kv_stage]);
            _kernels_data[kv_stage].kernels[0].skip_execution = impl_param._can_be_optimized || impl_param.state_layout.value().count() == 0;
        }
    }
};

namespace detail {

attach_kv_cache_impl::attach_kv_cache_impl() {
    auto types = { data_types::f16, data_types::f32 };
    auto formats = { format::bfyx };
    implementation_map<kv_cache>::add(impl_types::ocl,
                                           shape_types::dynamic_shape,
                                           kv_cache_impl::create,
                                           types,
                                           formats);

    implementation_map<kv_cache>::add(impl_types::ocl,
                                           shape_types::static_shape,
                                           kv_cache_impl::create,
                                           types,
                                           formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::kv_cache_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::kv_cache)
