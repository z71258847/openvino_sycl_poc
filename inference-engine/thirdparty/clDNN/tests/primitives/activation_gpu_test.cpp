// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/activation.hpp>
#include <cldnn/primitives/data.hpp>
#include <cldnn/primitives/reorder.hpp>

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

TEST(activation_gpu, basic_dynamic) {
    auto& engine = get_test_engine();

    layout in_actual_layout = {data_types::f32, format::bfyx, { 1, 1, 5, 4 }};
    layout in_dynamic_layout = {data_types::f32, format::bfyx, { 1, 1, dimension(1, 10), dimension(1, 10) }};
    auto input = engine.allocate_memory(in_actual_layout);
    set_values(input,
    { 1.0f, 0.0f, -3.0f, 4.0f, 5.0f,
      0.0f, 2.0f, 3.0f, 4.0f, -6.0f,
      3.0f, -3.0f, 3.0f, 0.0f, 1.0f,
      1.0f, 1.0f, 1.0f, -1.0f, 0.0f });
    VF<float> output_vec = {
        1.0f, 0.0f, 0.0f, 4.0f, 5.0f,
        0.0f, 2.0f, 3.0f, 4.0f, 0.0f,
        3.0f, 0.0f, 3.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 0.0f, 0.0f };

    topology topology(
        input_layout("input", in_dynamic_layout),
        activation("relu", "input", activation_func::relu));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.size.spatial(1);
    int x_size = output_layout.size.spatial(0);
    int f_size = output_layout.size.feature(0);
    int b_size = output_layout.size.batch(0);
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 10);
    EXPECT_EQ(x_size, 10);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(output_vec[i], output_ptr[i]) << " i=" << i;
    }
}
