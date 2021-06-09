// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/activation.hpp>

#include <cldnn/graph/topology.hpp>
#include <cldnn/graph/program.hpp>
#include <cldnn/graph/program_node.hpp>

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

struct activation_sp_test_params {
    std::vector<cldnn::layout> input_layouts;
    std::vector<cldnn::layout> expected_output_layout;
};

class activation_test : public ::testing::TestWithParam<activation_sp_test_params>{};

TEST_P(activation_test, basic) {
    auto params = GetParam();
    auto& engine = get_test_engine();

    ASSERT_EQ(params.input_layouts.size(), 1);
    ASSERT_EQ(params.expected_output_layout.size(), 1);

    topology topology(
        input_layout("input", params.input_layouts[0]),
        activation("relu", "input", activation_func::relu));

    build_options opts;
    program prog(engine, topology, opts, false, true);

    auto& node = prog.get_node("relu");

    // ASSERT_EQ(node.get_output_layouts().size(), 1);

    auto actual_layout = node.get_output_layout();
    auto expected_layout = params.expected_output_layout[0];

    ASSERT_EQ(actual_layout.size, expected_layout.size);
    ASSERT_EQ(actual_layout.data_type, expected_layout.data_type);
    ASSERT_EQ(actual_layout.format, expected_layout.format);
}


static std::vector<activation_sp_test_params> test_data = {
{{{data_types::f32, format::bfyx, {1, 2, 3, 4}}}, {{data_types::f32, format::bfyx, {1, 2, 3, 4}}}},
{{{data_types::f32, format::bfyx, {1, 2, 3, dimension::dynamic()}}}, {{data_types::f32, format::bfyx, {1, 2, 3, dimension::dynamic()}}}}
};

INSTANTIATE_TEST_SUITE_P(shape_infer, activation_test, ::testing::ValuesIn(test_data));
