// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_dpcpp_primitive_inst.h"
#include "intel_gpu/runtime/engine.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "jitter.h"
#include "register.hpp"

#include <map>
#include <sstream>
#include <vector>
#include <memory>
#include <string>

namespace cldnn {
namespace common {

struct custom_dpcpp_primitive_impl : typed_primitive_impl<custom_dpcpp_primitive> {
    const custom_dpcpp_primitive_node& outer;
    const custom_dpcpp_primitive::execute_function callback_function;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<custom_dpcpp_primitive_impl>(*this);
    }

    custom_dpcpp_primitive_impl(const custom_dpcpp_primitive_impl& other)
            : outer(other.outer)
            , callback_function(other.callback_function) {
    }

    custom_dpcpp_primitive_impl(const custom_dpcpp_primitive_node& arg,
                          const custom_dpcpp_primitive::execute_function& impl)
            : outer(arg)
            , callback_function(impl) {
    }

    void init_kernels() override { }

    void set_arguments_impl(custom_dpcpp_primitive_inst& instance) override {
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                            custom_dpcpp_primitive_inst& instance) override {
        std::vector<memory::ptr> inputs;
        inputs.reserve(instance.inputs_memory_count());
        for (auto& dep : instance.dependencies()) {
            inputs.push_back(dep->output_memory_ptr());
        }
        // TODO: support multiple outputs?
        std::vector<memory::ptr> outputs;
        outputs.push_back(instance.output_memory_ptr());

        auto& stream = instance.get_network().get_stream();


        return instance.node.get_primitive()->callback_function(stream, events, inputs, outputs);
    }
    static primitive_impl* create(const custom_dpcpp_primitive_node& arg) {
        const auto primitive = arg.get_primitive().get();
        return new custom_dpcpp_primitive_impl(arg, primitive->callback_function);
    }
};


namespace detail {
attach_custom_dpcpp_primitive_common::attach_custom_dpcpp_primitive_common() {
    implementation_map<custom_dpcpp_primitive>::add(impl_types::common, custom_dpcpp_primitive_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });
}

}  // namespace detail
}  // namespace common
}  // namespace cldnn
