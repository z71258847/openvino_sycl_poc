// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "slice_inst.h"
#include "implementation_map.hpp"

#include "intel_gpu/runtime/error_handler.hpp"

#include "openvino/op/slice.hpp"

namespace cldnn {
namespace cpu {

struct slice_impl : public typed_primitive_impl<slice> {
    using parent = typed_primitive_impl<slice>;
    using parent::parent;

    std::shared_ptr<ov::op::v8::Slice> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<slice_impl>(*this);
    }

    slice_impl() : parent("slice_cpu_impl") {}

    explicit slice_impl(const slice_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<slice>(), "[GPU] Incorrect program_node type");
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, slice_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "slice::execute_impl");
        auto& stream = instance.get_network().get_stream();

        for (auto e : events) {
            e->wait();
        }

        auto ev = stream.create_user_event(false);
        auto params = instance.get_impl_params();

        ov::TensorVector input_host_tensors;
        ov::TensorVector output_host_tensors;

        auto output_mem_ptr = instance.output_memory_ptr();

        for (size_t i = 0; i < params->input_layouts.size(); i++) {
            input_host_tensors.push_back(make_tensor(params->input_layouts[i], instance.dep_memory_ptr(i)->lock(stream, mem_lock_type::read)));
        }
        output_host_tensors.push_back(make_tensor(params->output_layouts[0], output_mem_ptr->lock(stream, mem_lock_type::write)));

        if (!op) {
            op = std::make_shared<ov::op::v8::Slice>();
        }

        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
                        "[GPU] Couldn't execute slice primitive with id ", instance.id());

        for (size_t i = 0; i < params->input_layouts.size(); i++) {
            instance.dep_memory_ptr(i)->unlock(stream);
        }

        output_mem_ptr->unlock(stream);

        ev->set();

        return ev;
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update_dispatch_data(const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const slice_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<slice_impl>();
    }
};


namespace detail {

attach_slice_impl::attach_slice_impl() {
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32,
        data_types::i64,
        data_types::i8,
        data_types::u8,
    };

    implementation_map<slice>::add(impl_types::cpu, shape_types::static_shape, slice_impl::create, types, formats);
    implementation_map<slice>::add(impl_types::cpu, shape_types::dynamic_shape, slice_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::slice_impl)
