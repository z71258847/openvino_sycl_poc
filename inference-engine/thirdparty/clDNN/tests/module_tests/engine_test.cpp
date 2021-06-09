// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "runtime/ocl/ocl_engine.hpp"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(engine_test, can_allocate_memory) {
    auto& engine = get_test_engine();
    layout l{data_types::i8, format::bfyx, {1}};
    memory::ptr mem_ptr = nullptr;
    ASSERT_NO_THROW(mem_ptr = engine.allocate_memory(l));
    ASSERT_TRUE(mem_ptr != nullptr);

    ASSERT_EQ(mem_ptr->count(), 1);
    ASSERT_EQ(mem_ptr->get_layout().data_type, data_types::i8);
    ASSERT_EQ(mem_ptr->get_layout().format, format::bfyx);
}

TEST(engine_test, cant_allocate_memory_for_dynamic_input) {
    auto& engine = get_test_engine();
    layout l{data_types::i8, format::bfyx, tensor::dynamic()};
    ASSERT_THROW(engine.allocate_memory(l), std::exception);
}

TEST(engine_test, can_allocate_memory_for_upper_bound) {
    auto& engine = get_test_engine();
    layout l{data_types::f32, format::bfyx, {1, dimension(1, 10)}};

    memory::ptr mem_ptr = nullptr;
    ASSERT_NO_THROW(mem_ptr = engine.allocate_memory(l.upper_bound()));
    ASSERT_TRUE(mem_ptr != nullptr);

    ASSERT_EQ(mem_ptr->count(), 10);
    ASSERT_EQ(mem_ptr->size(), 10 * sizeof(float));
    ASSERT_EQ(mem_ptr->get_layout().data_type, data_types::f32);
    ASSERT_EQ(mem_ptr->get_layout().format, format::bfyx);
}
