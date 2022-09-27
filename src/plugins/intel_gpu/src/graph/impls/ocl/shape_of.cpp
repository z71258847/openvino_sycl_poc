// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_of_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "shape_of/shape_of_kernel_selector.h"
#include "shape_of/shape_of_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct shape_of_impl : typed_primitive_impl_ocl<shape_of> {
    using parent = typed_primitive_impl_ocl<shape_of>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<shape_of_impl>(*this);
    }

    static std::pair<kernel_selector::shape_of_params, kernel_selector::shape_of_optional_params> get_params(const kernel_impl_params& impl_param) {
        auto shape_of_params = get_default_params<kernel_selector::shape_of_params>(impl_param);
        auto shape_of_optional_params =
            get_default_optional_params<kernel_selector::shape_of_optional_params>(impl_param.prog);

        auto input_layout = impl_param.input_layouts[0];
        shape_of_params.input_rank = input_layout.is_dynamic() ? input_layout.get_partial_shape().size() : input_layout.get_rank();
        shape_of_params.input_dims = input_layout.is_dynamic() ? std::vector<cldnn::tensor::value_type>{} : input_layout.get_dims();

        return {shape_of_params, shape_of_optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_params(impl_param);
        auto& kernel_data = this->_kernel_data;

        (kernel_data.update_kernels_func)(kernel_params.first, kernel_data);
    }

    static primitive_impl* create(const shape_of_node& arg, const kernel_impl_params& impl_param) {
        auto kernel_params = get_params(impl_param);
        auto& kernel_selector = kernel_selector::shape_of_instance();
        auto best_kernels = kernel_selector.GetBestKernels(kernel_params.first, kernel_params.second);
        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto shape_of = new shape_of_impl(arg, best_kernels[0]);

        return shape_of;
    }
};

namespace detail {

attach_shape_of_impl::attach_shape_of_impl() {
    implementation_map<shape_of>::add(impl_types::ocl, shape_of_impl::create, {shape_types::static_shape, shape_types::dynamic_shape}, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
