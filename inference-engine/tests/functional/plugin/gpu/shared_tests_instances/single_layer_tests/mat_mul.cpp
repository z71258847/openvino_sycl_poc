// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/mat_mul.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32
};

using input_shapes = std::pair<std::vector<size_t>, std::vector<size_t>>;

const std::vector<input_shapes> shapes = {
        { {1, 4, 5, 6}, {1, 4, 6, 4}},
        { {1, 4, 5, 6}, {6, 4}},
        { {1, 4, 7, 5, 6}, {1, 4, 1, 6, 4}},
        { {1, 5, 6}, {1, 6, 4}},
        { {7, 5, 6}, {7, 6, 4}},
        { {7, 5, 6}, {1, 6, 4}},
        { {10, 6}, {6, 4}},
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

INSTANTIATE_TEST_CASE_P(MatMul, MatMulTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(shapes),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
        MatMulTest::getTestCaseName);

} // namespace

