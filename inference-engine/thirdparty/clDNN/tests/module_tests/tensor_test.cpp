// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <cldnn/runtime/tensor.hpp>

using namespace cldnn;
using namespace ::tests;

TEST(tensor_test, can_create_static_tensor) {
    tensor a = {1, 2, 3, 4};
    ASSERT_EQ(a.rank().get_length(), 4);
    ASSERT_TRUE(a.is_static());
    ASSERT_EQ(a[0], 1);
    ASSERT_EQ(a[1], 2);
    ASSERT_EQ(a[2], 3);
    ASSERT_EQ(a[3], 4);
}

TEST(tensor_test, can_create_static_tensor_with_rank_and_fill_value) {
    tensor a(2, 3);
    ASSERT_EQ(a.rank().get_length(), 2);
    ASSERT_TRUE(a.is_static());
    ASSERT_EQ(a[0], 3);
    ASSERT_EQ(a[1], 3);
}

TEST(tensor_test, can_create_dynamic_tensor) {
    tensor a = {1, 2, 3, dimension::dynamic()};
    ASSERT_EQ(a.rank().get_length(), 4);
    ASSERT_TRUE(a.is_dynamic());
    ASSERT_EQ(a[0], 1);
    ASSERT_EQ(a[1], 2);
    ASSERT_EQ(a[2], 3);
    ASSERT_EQ(a[3], dimension::dynamic());
}

TEST(tensor_test, can_create_bounded_dynamic_tensor) {
    tensor a = {1, 2, 3, dimension(1, 10)};
    ASSERT_EQ(a.rank().get_length(), 4);
    ASSERT_TRUE(a.is_dynamic());
    ASSERT_EQ(a[0], 1);
    ASSERT_EQ(a[1], 2);
    ASSERT_EQ(a[2], 3);
    ASSERT_TRUE(a[3].is_dynamic());
    ASSERT_EQ(a[3].get_max_length(), 10);
    ASSERT_EQ(a[3].get_min_length(), 1);
}

TEST(tensor_test, max_static_rank) {
    tensor a = {1, 2, 3, 4};
    tensor b = {4, 3, 2, 1};

    auto c = tensor::max(a, b);

    ASSERT_EQ(c.rank().get_length(), 4);
    ASSERT_EQ(c[0], 4);
    ASSERT_EQ(c[1], 3);
    ASSERT_EQ(c[2], 3);
    ASSERT_EQ(c[3], 4);
}

TEST(tensor_test, max_exception_for_differnet_ranks) {
    tensor a = {1, 2, 3, 4};
    tensor b = {4, 3, 2};

    ASSERT_THROW(tensor::max(a, b), std::exception);
}

TEST(tensor_test, max_exception_for_dynamic_dims) {
    tensor a = {1, 2, 3, dimension::dynamic()};
    tensor b = {4, 3, 2, 1};

    ASSERT_THROW(tensor::max(a, b), std::exception);
}

TEST(tensor_test, min_static_rank) {
    tensor a = {1, 2, 3};
    tensor b = {4, 3, 2};

    auto c = tensor::min(a, b);

    ASSERT_EQ(c.rank().get_length(), 3);
    ASSERT_EQ(c[0], 1);
    ASSERT_EQ(c[1], 2);
    ASSERT_EQ(c[2], 2);
}

TEST(tensor_test, min_exception_for_differnet_ranks) {
    tensor a = {1, 2, 3, 4};
    tensor b = {4, 3, 2};

    ASSERT_THROW(tensor::min(a, b), std::exception);
}

TEST(tensor_test, min_exception_for_dynamic_dims) {
    tensor a = {1, 2, 3, dimension::dynamic()};
    tensor b = {4, 3, 2, 1};

    ASSERT_THROW(tensor::min(a, b), std::exception);
}

TEST(tensor_test, negate_static_rank) {
    tensor a = {1, 2, -3};

    auto c = a.negate();

    ASSERT_EQ(c.rank().get_length(), 3);
    ASSERT_EQ(c[0], -1);
    ASSERT_EQ(c[1], -2);
    ASSERT_EQ(c[2], 3);
}
