// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/runtime/format.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "intel_gpu/graph/primitive_impl.hpp"

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace ov {
namespace intel_gpu {

using Format = cldnn::format;
using ImplTypes = cldnn::impl_types;
struct FusedOpDesc {

};

/// A base class for templated GPUNode that maintains overridden input types and output types for an operation.
class OPENVINO_API GPUOpExtension {
public:
    virtual ~GPUOpExtension();

    GPUOpExtension() = default;
    GPUOpExtension(const element::TypeVector& _input_data_types, const element::TypeVector& _output_data_types) {}

    void visit_attributes(AttributeVisitor& visitor);

    void set_preferred_impl_type(ImplTypes impl) { m_impl_type = impl; }
    ImplTypes get_preferred_impl_type() const { return m_impl_type; }

    std::vector<Format> get_preferred_input_fmts() const;
    std::vector<Format> get_preferred_output_fmts() const;

    std::vector<Format>& get_preferred_input_fmts();
    std::vector<Format>& get_preferred_output_fmts();

    std::vector<Format> get_preferred_input_fmts(ImplTypes impl_type) const;
    std::vector<Format> get_preferred_output_fmts(ImplTypes impl_type) const;

    std::vector<Format>& get_preferred_input_fmts(ImplTypes impl_type);
    std::vector<Format>& get_preferred_output_fmts(ImplTypes impl_type);

    Format get_preferred_input_fmt(size_t idx = 0) const;
    Format get_preferred_output_fmt(size_t idx = 0) const;

    void set_preferred_input_fmt(size_t idx, Format type);
    void set_preferred_output_fmt(size_t idx, Format type);

    Format get_preferred_input_fmt(ImplTypes impl_type, size_t idx = 0) const;
    Format get_preferred_output_fmt(ImplTypes impl_type, size_t idx = 0) const;

    void set_preferred_input_fmts(ImplTypes impl_type, std::vector<Format> fmts);
    void set_preferred_output_fmts(ImplTypes impl_type, std::vector<Format> fmts);

    void copy_preferred_params(const GPUOpExtension& other);
    void copy_preferred_output_fmts(const GPUOpExtension& other);
    void copy_preferred_input_fmts(const GPUOpExtension& other);

    std::set<ImplTypes> get_available_impl_types() const;
    void set_available_impl_types(const std::set<ImplTypes>& impls);

    void set_implementation(std::unique_ptr<cldnn::primitive_impl> impl);

    // void add_fused_ops();

protected:
    // size_t unique_id = 0;
    // static thread_local size_t cur_id;

    std::set<ImplTypes> m_available_impl_types;
    ImplTypes m_impl_type = ImplTypes::any;

    std::map<ImplTypes, std::vector<Format>> m_preferred_input_fmts;
    std::map<ImplTypes, std::vector<Format>> m_preferred_output_fmts;

    std::unique_ptr<cldnn::primitive_impl> m_selected_impl;
    // list of primitives that can reuse same memory buffers due to execution order conflicts
    // std::set<primitive_id> memory_dependencies;

    bool m_optimized = false;
    bool m_share_buffer = true;

    // std::vector<FusedOpDesc> m_fused_ops;
    std::shared_ptr<ov::Model> m_fused_ops = nullptr;
};

}  // namespace op
}  // namespace ov
