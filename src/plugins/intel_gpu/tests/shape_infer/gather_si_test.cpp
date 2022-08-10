// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "gather_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;


#if 1
#define PRINT_TIME(func) \
{ \
 auto start = std::chrono::high_resolution_clock::now(); \
 func; \
 auto duration = std::chrono::high_resolution_clock::now() - start; \
 std::cerr <<  #func <<  " " << std::chrono::duration_cast<std::chrono::microseconds>(duration).count() << "us\n"; \
}
#else
#define PRINT_TIME(func) func;
#endif


namespace shape_infer_tests {

struct gather_test_params {
    layout in0_layout;
    layout in1_layout;
    int64_t axis;
    int64_t batch_dim;
    layout expected_layout;
};

class gather_test : public testing::TestWithParam<gather_test_params> { };

TEST_P(gather_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input0_layout_prim = std::make_shared<input_layout>("input0", p.in0_layout);
    auto input1_layout_prim = std::make_shared<input_layout>("input1", p.in1_layout);
    auto gather_prim = std::make_shared<gather>("output", "input0", "input1", p.axis, ov::Shape{}, p.batch_dim);

    cldnn::program prog(engine);

    auto& input0_layout_node = prog.get_or_create(input0_layout_prim);
    auto& input1_layout_node = prog.get_or_create(input1_layout_prim);
    auto& gather_node = prog.get_or_create(gather_prim);
    program_wrapper::add_connection(prog, input0_layout_node, gather_node);
    program_wrapper::add_connection(prog, input1_layout_node, gather_node);
    // auto res = gather_inst::calc_output_layouts<ov::PartialShape>(gather_node, *gather_node.get_kernel_impl_params());

    std::vector<layout> results;
    std::vector<layout> results1;

    results.reserve(1000);
    results1.reserve(1000);
    PRINT_TIME(
        for (size_t i = 0; i < 1000; i++) {
            results.push_back(gather_inst::calc_output_layouts<ov::PartialShape>(gather_node, *gather_node.get_kernel_impl_params())[0]);

        }
    );

    PRINT_TIME(
        for (size_t i = 0; i < 1000; i++) {
            results1.push_back(gather_inst::calc_output_layouts<ov::intel_gpu::StaticShape>(gather_node, *gather_node.get_kernel_impl_params())[0]);
        }
    );

    ASSERT_EQ(results[0], p.expected_layout);
    ASSERT_EQ(results1[0], p.expected_layout);

    // ASSERT_EQ(res.size(), 1);
    // ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, gather_test,
    testing::ValuesIn(std::vector<gather_test_params>{
        {
            layout{ov::PartialShape{1, 2, 3}, data_types::f32, format::bfyx}, layout{ov::PartialShape{4, 5}, data_types::f32, format::bfyx},
            1, 0,
            layout{ov::PartialShape{1, 4, 5, 3}, data_types::f32, format::bfyx}
        },
    }));

}  // shape_infer_tests
