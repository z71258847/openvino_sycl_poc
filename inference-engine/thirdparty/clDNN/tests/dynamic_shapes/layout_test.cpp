// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <cldnn/runtime/layout.hpp>

using namespace cldnn;
using namespace ::tests;

TEST(layout_test, can_create_layout_with_static_shape_rank_4) {
    tensor size = {1, 2, 3, 4};
    layout layout = {data_types::f32, format::bfyx, size};

    ASSERT_EQ(layout.data_type, data_types::f32);
    ASSERT_EQ(layout.format, format::bfyx);
    ASSERT_FALSE(layout.is_dynamic());
    ASSERT_FALSE(layout.size.is_dynamic());
    ASSERT_EQ(layout.size.rank(), 4);
    ASSERT_EQ(layout.size, size);
    ASSERT_EQ(layout.count(), 24);
}

TEST(layout_test, can_create_layout_with_static_shape_rank_2) {
    tensor size = {1, 2};
    layout layout = {data_types::f32, format::bfyx, size};

    ASSERT_EQ(layout.data_type, data_types::f32);
    ASSERT_EQ(layout.format, format::bfyx);
    ASSERT_FALSE(layout.is_dynamic());
    ASSERT_FALSE(layout.size.is_dynamic());
    ASSERT_EQ(layout.size.rank(), 2);
    ASSERT_EQ(layout.size, size);
    ASSERT_EQ(layout.count(), 2);
}

TEST(layout_test, can_create_layout_with_dynamic_shape_rank_4) {
    tensor size = {1, 2, dimension::dynamic(), dimension::dynamic()};
    layout layout = {data_types::f32, format::bfyx, size};

    ASSERT_EQ(layout.data_type, data_types::f32);
    ASSERT_EQ(layout.format, format::bfyx);
    ASSERT_TRUE(layout.is_dynamic());
    ASSERT_TRUE(layout.size.is_dynamic());
    ASSERT_EQ(layout.size.rank(), 4);
    ASSERT_EQ(layout.size, size);
    ASSERT_THROW(layout.count(), std::exception);
}

TEST(layout_test, can_create_layout_with_dynamic_shape_dynamic_rank) {
    tensor size = tensor::dynamic();
    layout layout = {data_types::f32, format::bfyx, size};

    ASSERT_EQ(layout.data_type, data_types::f32);
    ASSERT_EQ(layout.format, format::bfyx);
    ASSERT_TRUE(layout.is_dynamic());
    ASSERT_TRUE(layout.size.is_dynamic());
    ASSERT_TRUE(layout.size.rank().is_dynamic());
    ASSERT_EQ(layout.size, size);
    ASSERT_THROW(layout.count(), std::exception);
}
