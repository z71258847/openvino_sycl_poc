// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "openvino/op/constant.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace ngraph;
using namespace ov::test;

namespace SubgraphTestsDefinitions {
/*
 *                        Subtract_const(U8)
 *                           /
 *    Weights(NF4)       Convert(F32)
 *       |               /
 *    Convert(F32)      /
 *            \        /       Multiply_const(F32)
 *            Subtract(optional)     /
 *                  \               /
 *                   \             /
 *                       Multiply
 *                         |
 *      Data(F32)   Reshape(optional)
 *            \     /
 *             Matmul
 *               |
 *              Bias
 */
using MatmulWeightsDecompressionNF4Params = std::tuple<std::vector<InputShape>,  // input shapes
                                                    ov::test::ElementType,    // weights precision
                                                    ov::test::ElementType,    // activations precision
                                                    bool,                     // decompression subtract
                                                    bool,                     // reshape on decompression constants
                                                    std::map<std::string, std::string>>;  // additional config

class MatmulWeightsDecompressionNF4 : public testing::WithParamInterface<MatmulWeightsDecompressionNF4Params>, public SubgraphBaseTest {
public:
    static std::string get_test_case_name(testing::TestParamInfo<MatmulWeightsDecompressionNF4Params> obj) {
        std::vector<InputShape> inputShapes;
        ov::test::ElementType weights_precision;
        ov::test::ElementType activations_precision;
        bool decompression_sub;
        bool reshape_on_decompression;
        std::map<std::string, std::string> additional_config;

        std::tie(inputShapes,
                 weights_precision,
                 activations_precision,
                 decompression_sub,
                 reshape_on_decompression,
                 additional_config) = obj.param;

        std::ostringstream result;
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "weights_precision=" << weights_precision << "_";
        result << "activations_precision=" << activations_precision << "_";
        result << "decompression_subtract=" << decompression_sub << "_";
        result << "reshape_on_decompression=" << reshape_on_decompression << "_";

        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << ", " << configEntry.second << ":";
        }
        result << ")";

        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph(std::vector<ov::PartialShape>& inputShapes,
                                             const ov::element::Type data_precision,
                                             const ov::element::Type weights_precision,
                                             const bool add_subtract,
                                             const bool reshape_on_decompression) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(data_precision, inputShapes[0])};

        auto weights_shape = inputShapes[1].to_shape();
        auto weights_data = ov::test::utils::create_and_fill_tensor(weights_precision, weights_shape);
        auto weights = std::make_shared<ov::op::v0::Constant>(weights_data);
        weights->set_friendly_name("Compressed_weights");
        auto weights_convert = std::make_shared<ngraph::opset1::Convert>(weights, data_precision);

        auto scaleshift_target_shape = weights_shape;
        scaleshift_target_shape.back() = 1;

        auto matmul_weights_shape = { weights_shape[0], weights_shape[1] * weights_shape[2] };
        // if (add_subtract) {
        //     auto shift_const = ngraph::builder::makeConstant<uint8_t>(weights_precision, scaleshift_const_shape, {}, true);
        //     std::shared_ptr<ov::Node> shift_convert = std::make_shared<ngraph::opset1::Convert>(shift_const, data_precision);
        //     if (reshape_on_decompression) {
        //         auto shift_reshape_const = ov::opset10::Constant::create(ov::element::i32, {scaleshift_target_shape.size()}, scaleshift_target_shape);
        //         auto shift_reshape = std::make_shared<ov::opset10::Reshape>(shift_convert, shift_reshape_const, false);
        //         shift_convert = shift_reshape;
        //     }
        //     mul_parent = std::make_shared<ov::opset10::Subtract>(weights_convert, shift_convert);
        // }

        auto scale_const = ngraph::builder::makeConstant<float>(data_precision, scaleshift_target_shape, {}, true);
        auto multiply = std::make_shared<ov::opset10::Multiply>(weights_convert, scale_const);
        std::shared_ptr<ov::Node> weights_input = multiply;
        if (reshape_on_decompression) {
            auto reshape_const = ov::opset10::Constant::create(ov::element::i32, {matmul_weights_shape.size()}, matmul_weights_shape);
            auto reshape = std::make_shared<ov::opset10::Reshape>(multiply, reshape_const, false);
            weights_input = reshape;
        }

        auto matMul = builder::makeMatMul(params[0], weights_input, false, true);
        return std::make_shared<ov::Model>(NodeVector{matMul}, params, "MatmulWeightsDecompressionNF4");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        std::vector<InputShape> inputShapes;
        ov::test::ElementType weights_precision;
        ov::test::ElementType activations_precision;
        bool decompression_sub;
        bool reshape_on_decompression;
        std::map<std::string, std::string> additional_config;

        std::tie(inputShapes,
                 weights_precision,
                 activations_precision,
                 decompression_sub,
                 reshape_on_decompression,
                 additional_config) = GetParam();

        configuration.insert(additional_config.begin(), additional_config.end());
        init_input_shapes(inputShapes);

        inType = outType = activations_precision;

        function = init_subgraph(inputDynamicShapes, activations_precision, weights_precision, decompression_sub, reshape_on_decompression);
    }

    void checkResults() {
        const auto& test_param = GetParam();
        ov::test::ElementType weights_precision = std::get<1>(test_param);
        for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
            if (n->get_friendly_name() == "Compressed_weights") {
                ASSERT_EQ(n->get_output_element_type(0), weights_precision);
            }
        }
    }
};

TEST_P(MatmulWeightsDecompressionNF4, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    checkResults();
}

namespace {

const std::vector<ov::test::ElementType> activations_precisions = {ov::element::f32, ov::element::f16};
const std::vector<ov::test::ElementType> weights_precisions = {ov::element::nf4};
const std::vector<std::vector<InputShape>> input_shapes_basic = {
    {{{-1,-1,64}, {{1, 1, 64} , {10, 16, 64}}}, {{}, {{64,32,2}}}},
    {{{-1,-1,4096}, {{1, 4, 4096}, {10, 16, 4096}}}, {{}, {{4096,32,128}}}},
    // {{{}, {{1, 4, 16}}}, {{}, {{1, 16, 32}}}},
    // {{{}, {{10, 40, 496}}}, {{}, {{1, 496, 240}}}},
    // {{{}, {{1, 4, 48}}}, {{}, {{48, 256}}}},
    // {{{}, {{11, 339, 377}}}, {{}, {{377, 335}}}},
    // {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {{}, {{1, 480, 256}}}},
    // {{{}, {{1, 4, 32}}}, {{}, {{32, 256}}}},
    // {{{}, {{1, 4, 512}}}, {{}, {{512, 256}}}},
    // {{{}, {{1, 16, 32}}}, {{}, {{32, 64}}}},
    // {{{}, {{2, 4, 32}}}, {{}, {{32, 65}}}},
    // {{{}, {{3, 12, 768}}}, {{}, {{768, 1024}}}},
    // {{{}, {{11, 339, 577}}}, {{}, {{577, 335}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_basic,
                         MatmulWeightsDecompressionNF4,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(activations_precisions),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(std::map<std::string, std::string>())),
                         MatmulWeightsDecompressionNF4::get_test_case_name);

// const std::vector<std::vector<InputShape>> input_shapes_corner_cases_basic = {
//     {{{-1, -1, -1}, {{1, 4, 16}}}, {{}, {{1, 16, 32}}}},
//     {{{-1, -1, -1}, {{1, 4, 16}}}, {{}, {{16, 32}}}},
// };
// const std::vector<std::vector<InputShape>> input_shapes_corner_cases_big = {
//     {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {{}, {{1, 480, 256}}}},
// };

// const std::vector<bool> transpose_weights = {true, false};
// const std::vector<bool> add_decompression_sub = {true, false};
// const std::vector<bool> reshape_on_decompression = {true, false};

// INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_basic,
//                          MatmulWeightsDecompressionNF4,
//                          ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_basic),
//                                             ::testing::ValuesIn(weights_precisions),
//                                             ::testing::ValuesIn(activations_precisions),
//                                             ::testing::ValuesIn(transpose_weights),
//                                             ::testing::ValuesIn(add_decompression_sub),
//                                             ::testing::ValuesIn(reshape_on_decompression),
//                                             ::testing::Values(std::map<std::string, std::string>{})),
//                          MatmulWeightsDecompressionNF4::get_test_case_name);

// INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_big,
//                          MatmulWeightsDecompressionNF4,
//                          ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_big),
//                                             ::testing::ValuesIn(weights_precisions),
//                                             ::testing::ValuesIn(activations_precisions),
//                                             ::testing::ValuesIn(transpose_weights),
//                                             ::testing::ValuesIn(add_decompression_sub),
//                                             ::testing::ValuesIn(reshape_on_decompression),
//                                             ::testing::Values(std::map<std::string, std::string>{})),
//                          MatmulWeightsDecompressionNF4::get_test_case_name);
} // namespace

} // namespace SubgraphTestsDefinitions
