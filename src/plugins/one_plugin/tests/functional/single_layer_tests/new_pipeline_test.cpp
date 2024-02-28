// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include "common_test_utils/node_builders/fully_connected.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_op/convolution.hpp"
#include "common_test_utils/node_builders/convolution.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {
using ov::test::InputShape;

class NewPipelineTest : public testing::WithParamInterface<int>,
                                virtual public ov::test::SubgraphBaseTest {
protected:
    void SetUp() override {
        InputShape inputShape = ov::test::static_shapes_to_test_representation({{1, 3, 10, 11}})[0];
        init_input_shapes({inputShape});

        ov::op::PadType padType = ov::op::PadType::AUTO;
        ov::Strides kernel = {3, 3};
        ov::Strides stride = {1, 1};
        ov::Strides dilation = {1, 1};
        ov::CoordinateDiff padBegin = {0, 0};
        ov::CoordinateDiff padEnd = {0, 0};
        size_t out_channels = 8;
        ov::element::Type et = ov::element::f32;

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes)
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(et, shape));

        this->targetDevice = "ONE";

        auto convolutionNode = ov::test::utils::make_convolution(inputParams.front(), et, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, out_channels);

        auto relu = std::make_shared<ov::op::v0::Relu>(convolutionNode);
        auto target_shape = ov::op::v0::Constant::create(ov::element::i32, {2}, std::vector<int32_t>{8, 110});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(relu, target_shape, false);
        auto fc = ov::test::utils::make_fully_connected(reshape, et, 16, false, ov::Shape{});
        auto abs = std::make_shared<ov::op::v0::Abs>(fc);

        function = std::make_shared<ov::Model>(ov::NodeVector{abs}, inputParams, "NewPipelineTest");
    }
};

TEST_P(NewPipelineTest, Inference) {
    compile_model();
}

INSTANTIATE_TEST_SUITE_P(smoke, NewPipelineTest, ::testing::Values(0));
}  // namespace
