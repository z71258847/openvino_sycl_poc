// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/primitive.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/lru_cache.hpp"
#include "intel_gpu/runtime/tensor_accessor.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/helpers.hpp"
#include "intel_gpu/graph/serialization/polymorphic_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/layout_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "runtime/kernels_cache.hpp"

#include <memory>
#include <vector>
#include <string>

namespace cldnn {

class primitive_inst;
struct program_node;

/*
    Base class for all implementations.
*/
struct primitive_impl {
    primitive_impl() = default;
    explicit primitive_impl(const std::shared_ptr<WeightsReorderParams>& params, std::string kernel_name = "", bool is_dynamic = false)
        : _weights_reorder_params(params), _kernel_name(kernel_name), _is_dynamic(is_dynamic) {
    }
    explicit primitive_impl(std::string kernel_name, bool is_dynamic = false) :
        primitive_impl(nullptr, std::move(kernel_name), is_dynamic) {}
    virtual ~primitive_impl() = default;

    virtual std::vector<layout> get_internal_buffer_layouts() const = 0;
    virtual void set_node_params(const program_node&) {}
    virtual const std::string& get_type_info() const = 0;
    virtual void set_arguments(primitive_inst& instance) = 0;
    virtual void set_arguments(primitive_inst& instance, kernel_arguments_data& args) = 0;
    virtual event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) = 0;
    std::string get_kernel_name() const { return _kernel_name; }

    // class typed_primitive_gpu_impl override this with return false;
    virtual bool is_cpu() const { return true; }
    virtual bool is_onednn() const { return false; }
    virtual void init_kernels(const kernels_cache& kernels_cache, const kernel_impl_params& params) = 0;
    virtual void init_by_cached_kernels(const kernels_cache&, std::vector<std::string>& cached_kernel_ids) {}
    virtual std::vector<std::string> get_cached_kernel_ids(const kernels_cache&) { return {}; }
    virtual std::unique_ptr<primitive_impl> clone() const = 0;
    virtual std::vector<std::shared_ptr<cldnn::kernel_string>> get_kernels_source() { return {}; }
    virtual void reset_kernels_source() {}
    virtual std::vector<kernel::ptr> get_kernels() const { return {}; }
    virtual void save(cldnn::BinaryOutputBuffer& ob) const {
        ob << can_reuse_memory;
        ob << _kernel_name;
        ob << _is_dynamic;
        if (_weights_reorder_params == nullptr) {
            ob << false;
        } else {
            ob << true;
            _weights_reorder_params->save(ob);
        }
    }
    virtual void load(cldnn::BinaryInputBuffer& ib) {
        ib >> can_reuse_memory;
        ib >> _kernel_name;
        ib >> _is_dynamic;
        bool has_weights_reorder_params;
        ib >> has_weights_reorder_params;
        if (has_weights_reorder_params) {
            _weights_reorder_params = std::make_shared<WeightsReorderParams>();
            _weights_reorder_params->load(ib);
        }
    }
    // returns a pair of batch program hash and kernel entry of each ocl impl. Returns "" for other impl types.
    virtual std::pair<std::string, std::string> get_kernels_dump_info() const {
        return std::make_pair("", "");
    }

    // If this flag is set as false, the memory allocated for this primitive is not allowed to be reused
    bool can_reuse_memory = true;

    void set_dynamic(bool val) { _is_dynamic = val; }
    bool is_dynamic() const { return _is_dynamic; }

    virtual void update_dispatch_data(const kernel_impl_params& impl_params) {
        OPENVINO_ASSERT(_is_dynamic, "[GPU] update_dispatch_data is called for static shape implementation ", _kernel_name);
        OPENVINO_ASSERT(false, "[GPU] update_dispatch_data is not implemented for dynamic implemenation ", _kernel_name);
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params);

    virtual kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const {
        return primitive_impl::static_canonicalize_shapes(impl_params);
    }

    virtual void set_kernels(cldnn::kernels_cache::compiled_kernels kernels) {}
    virtual std::vector<kernel::ptr> get_kernels() { return {}; }

    bool need_weights_reorder() const { return _weights_reorder_params != nullptr; }
    std::shared_ptr<WeightsReorderParams> get_weights_reorder_params() const { return _weights_reorder_params; }

    std::shared_ptr<kernel_impl_params> get_weights_reorder_kernel_params() const;

protected:
    std::shared_ptr<WeightsReorderParams> _weights_reorder_params = nullptr;
    std::string _kernel_name;
    bool _is_dynamic = false;
};

}  // namespace cldnn
