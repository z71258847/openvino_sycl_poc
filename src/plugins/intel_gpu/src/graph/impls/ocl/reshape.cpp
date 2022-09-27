// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "reshape/reshape_kernel_ref.h"
#include "reshape/reshape_kernel_selector.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct reshape_impl : public typed_primitive_impl_ocl<reshape> {
    using parent = typed_primitive_impl_ocl<reshape>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reshape_impl>(*this);
    }

public:
    static std::pair<kernel_selector::reshape_params, kernel_selector::reshape_optional_params> get_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<reshape>();
        auto reshape_params = get_default_params<kernel_selector::reshape_params>(impl_param);
        auto reshape_optional_params =
            get_default_optional_params<kernel_selector::reshape_optional_params>(impl_param.prog);

        return {reshape_params, reshape_optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        if (this->_kernel_data.kernels.empty())
            return;

        auto kernel_params = get_params(impl_param);
        auto& kernel_data = this->_kernel_data;

        (kernel_data.update_kernels_func)(kernel_params.first, kernel_data);
    }

    static primitive_impl* create(reshape_node const& arg, const kernel_impl_params& impl_param) {
        if (arg.can_be_optimized()) {
            return new reshape_impl(arg, {});
        }

        auto kernel_params = get_params(impl_param);

        auto& kernel_selector = kernel_selector::reshape_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(kernel_params.first, kernel_params.second);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto reshape = new reshape_impl(arg, best_kernels[0]);

        return reshape;
    }
};

namespace detail {

attach_reshape_impl::attach_reshape_impl() {
    implementation_map<reshape>::add(impl_types::ocl, reshape_impl::create, {shape_types::static_shape, shape_types::dynamic_shape}, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
