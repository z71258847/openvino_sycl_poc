// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_impl_params.hpp"

#include "intel_gpu/graph/program.hpp"

#include "intel_gpu/graph/serialization/layout_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include <string>
#include <vector>

namespace cldnn {

void kernel_impl_params::save(BinaryOutputBuffer& ob) const {
    ob << unique_id;
    ob << input_layouts;
    ob << output_layouts;
    ob << input_offsets.size();
    for (size_t i = 0; i < input_offsets.size(); i++) {
        ob << input_offsets[i].sizes();
    }

    if (weights_layout.has_value()) {
        ob << true;
        ob << weights_layout.value();
    } else {
        ob << false;
    }

    if (bias_layout.has_value()) {
        ob << true;
        ob << bias_layout.value();
    } else {
        ob << false;
    }

    if (weights_zero_points_layout.has_value()) {
        ob << true;
        ob << weights_zero_points_layout.value();
    } else {
        ob << false;
    }

    if (activations_zero_points_layout.has_value()) {
        ob << true;
        ob << activations_zero_points_layout.value();
    } else {
        ob << false;
    }

    if (compensation_layout.has_value()) {
        ob << true;
        ob << compensation_layout.value();
    } else {
        ob << false;
    }

    ob << fused_desc.size();
#ifdef ENABLE_ONEDNN_FOR_GPU
    size_t num_fused_prims = fused_desc_onednn.size();
    ob << num_fused_prims;
    for (auto fused_prim : fused_desc_onednn) {
        ob << make_data(&fused_prim, sizeof(fused_primitive_desc_onednn));
    }
#endif // ENABLE_ONEDNN_FOR_GPU
    ob << primary_input_idx;
}

void kernel_impl_params::load(BinaryInputBuffer& ib) {
    prog = nullptr;
    desc = nullptr;
    ib >> unique_id;
    ib >> input_layouts;
    ib >> output_layouts;
    {
        size_t num_input_offsets;
        ib >> num_input_offsets;
        input_offsets.resize(num_input_offsets);
        for (size_t i = 0; i < num_input_offsets; i++) {
            std::vector<cldnn::tensor::value_type> sizes;
            ib >> sizes;
            input_offsets[i] = cldnn::tensor(sizes);
        }
    }
    bool has_value = false;
    layout layout_buf;

    ib >> has_value;
    if (has_value) {
        ib >> layout_buf;
        weights_layout = layout_buf;
    }

    ib >> has_value;
    if (has_value) {
        ib >> layout_buf;
        bias_layout = layout_buf;
    }

    ib >> has_value;
    if (has_value) {
        ib >> layout_buf;
        weights_zero_points_layout = layout_buf;
    }

    ib >> has_value;
    if (has_value) {
        ib >> layout_buf;
        activations_zero_points_layout = layout_buf;
    }

    ib >> has_value;
    if (has_value) {
        ib >> layout_buf;
        compensation_layout = layout_buf;
    }

    {
        // Fake fused_desc just for has_fused_primitives()
        size_t num_fused_desc;
        ib >> num_fused_desc;
        if (num_fused_desc > 0) {
            fused_desc.emplace_back(cldnn::fused_primitive_desc(nullptr));
        }
    }
#ifdef ENABLE_ONEDNN_FOR_GPU
    size_t num_fused_prims;
    ib >> num_fused_prims;
    fused_desc_onednn.resize(num_fused_prims);
    for (size_t idx = 0; idx < num_fused_prims; ++idx) {
        ib >> make_data(&fused_desc_onednn[idx], sizeof(fused_primitive_desc_onednn));
    }
#endif // ENABLE_ONEDNN_FOR_GPU
    ib >> primary_input_idx;
}

}  // namespace cldnn
