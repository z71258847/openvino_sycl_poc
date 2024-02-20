// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _OPENVINO_OP_REG
#    warning "_OPENVINO_OP_REG not defined"
#    define _OPENVINO_OP_REG(x, y)
#endif

_OPENVINO_OP_REG(Abs_v0, ov::op::v0::Abs);
_OPENVINO_OP_REG(Asin_v0, ov::op::v0::Asin);
_OPENVINO_OP_REG(Acos_v0, ov::op::v0::Acos);
_OPENVINO_OP_REG(Atan_v0, ov::op::v0::Atan);
_OPENVINO_OP_REG(Ceiling_v0, ov::op::v0::Ceiling);
_OPENVINO_OP_REG(Clamp_v0, ov::op::v0::Clamp);
_OPENVINO_OP_REG(Concat_v0, ov::op::v0::Concat);
_OPENVINO_OP_REG(Constant_v0, ov::op::v0::Constant);
_OPENVINO_OP_REG(Convert_v0, ov::op::v0::Convert);
_OPENVINO_OP_REG(Cos_v0, ov::op::v0::Cos);
_OPENVINO_OP_REG(Cosh_v0, ov::op::v0::Cosh);
_OPENVINO_OP_REG(CumSum_v0, ov::op::v0::CumSum);
_OPENVINO_OP_REG(CTCGreedyDecoder_v0, ov::op::v0::CTCGreedyDecoder);
_OPENVINO_OP_REG(DepthToSpace_v0, ov::op::v0::DepthToSpace);
_OPENVINO_OP_REG(DetectionOutput_v0, ov::op::v0::DetectionOutput);
_OPENVINO_OP_REG(Elu_v0, ov::op::v0::Elu);
_OPENVINO_OP_REG(Erf_v0, ov::op::v0::Erf);
_OPENVINO_OP_REG(Exp_v0, ov::op::v0::Exp);
_OPENVINO_OP_REG(FakeQuantize_v0, ov::op::v0::FakeQuantize);
_OPENVINO_OP_REG(Floor_v0, ov::op::v0::Floor);
_OPENVINO_OP_REG(Gelu_v0, ov::op::v0::Gelu);
_OPENVINO_OP_REG(GRN_v0, ov::op::v0::GRN);
_OPENVINO_OP_REG(HardSigmoid_v0, ov::op::v0::HardSigmoid);
// _OPENVINO_OP_REG(Interpolate_v0, ov::op::v0::Interpolate); Supported via v0 -> v4 conversion
_OPENVINO_OP_REG(Log_v0, ov::op::v0::Log);
_OPENVINO_OP_REG(LRN_v0, ov::op::v0::LRN);
_OPENVINO_OP_REG(MatMul_v0, ov::op::v0::MatMul);
_OPENVINO_OP_REG(MVN_v0, ov::op::v0::MVN);
_OPENVINO_OP_REG(Negative_v0, ov::op::v0::Negative);
_OPENVINO_OP_REG(NormalizeL2_v0, ov::op::v0::NormalizeL2);
_OPENVINO_OP_REG(Parameter_v0, ov::op::v0::Parameter);
_OPENVINO_OP_REG(PRelu_v0, ov::op::v0::PRelu);
_OPENVINO_OP_REG(PriorBox_v0, ov::op::v0::PriorBox);
_OPENVINO_OP_REG(PriorBoxClustered_v0, ov::op::v0::PriorBoxClustered);
_OPENVINO_OP_REG(Proposal_v0, ov::op::v0::Proposal);
_OPENVINO_OP_REG(PSROIPooling_v0, ov::op::v0::PSROIPooling);
_OPENVINO_OP_REG(Relu_v0, ov::op::v0::Relu);
_OPENVINO_OP_REG(Result_v0, ov::op::v0::Result);
_OPENVINO_OP_REG(RegionYolo_v0, ov::op::v0::RegionYolo);
_OPENVINO_OP_REG(ReorgYolo_v0, ov::op::v0::ReorgYolo);
_OPENVINO_OP_REG(ReverseSequence_v0, ov::op::v0::ReverseSequence);
_OPENVINO_OP_REG(ROIPooling_v0, ov::op::v0::ROIPooling);
_OPENVINO_OP_REG(Sigmoid_v0, ov::op::v0::Sigmoid);
_OPENVINO_OP_REG(Sqrt_v0, ov::op::v0::Sqrt);
_OPENVINO_OP_REG(Selu_v0, ov::op::v0::Selu);
_OPENVINO_OP_REG(Sin_v0, ov::op::v0::Sin);
_OPENVINO_OP_REG(Sinh_v0, ov::op::v0::Sinh);
_OPENVINO_OP_REG(Sign_v0, ov::op::v0::Sign);
_OPENVINO_OP_REG(SquaredDifference_v0, ov::op::v0::SquaredDifference);
_OPENVINO_OP_REG(SpaceToDepth_v0, ov::op::v0::SpaceToDepth);
_OPENVINO_OP_REG(Squeeze_v0, ov::op::v0::Squeeze);
_OPENVINO_OP_REG(ShapeOf_v0, ov::op::v0::ShapeOf);
_OPENVINO_OP_REG(ShuffleChannels_v0, ov::op::v0::ShuffleChannels);
_OPENVINO_OP_REG(Tan_v0, ov::op::v0::Tan);
_OPENVINO_OP_REG(Tanh_v0, ov::op::v0::Tanh);
// _OPENVINO_OP_REG(TensorIterator_v0, ov::op::v0::TensorIterator);
_OPENVINO_OP_REG(Tile_v0, ov::op::v0::Tile);
_OPENVINO_OP_REG(Unsqueeze_v0, ov::op::v0::Unsqueeze);

// ----------------------------- Unsupported v0 ops ----------------------------- //
// Deprecated ops
// _OPENVINO_OP_REG(Add_v0, ov::op::v0::Add);
// _OPENVINO_OP_REG(Divide_v0, ov::op::v0::Divide);
// _OPENVINO_OP_REG(Greater_v0, ov::op::v0::Greater);
// _OPENVINO_OP_REG(GreaterEq_v0, ov::op::v0::GreaterEq);
// _OPENVINO_OP_REG(Less_v0, ov::op::v0::Less);
// _OPENVINO_OP_REG(LessEq_v0, ov::op::v0::LessEq);
// _OPENVINO_OP_REG(LSTMSequence_v0, ov::op::v0::LSTMSequence);
// _OPENVINO_OP_REG(LSTMCell_v0, ov::op::v0::LSTMCell);
// _OPENVINO_OP_REG(Maximum_v0, ov::op::v0::Maximum);
// _OPENVINO_OP_REG(Minimum_v0, ov::op::v0::Minimum);
// _OPENVINO_OP_REG(Multiply_v0, ov::op::v0::Multiply);
// _OPENVINO_OP_REG(NotEqual_v0, ov::op::v0::NotEqual);
// _OPENVINO_OP_REG(Power_v0, ov::op::v0::Power);
// _OPENVINO_OP_REG(Quantize_v0, ov::op::v0::Quantize);
// _OPENVINO_OP_REG(Select_v0, ov::op::v0::Select);
// _OPENVINO_OP_REG(Subtract_v0, ov::op::v0::Subtract);
// _OPENVINO_OP_REG(Xor_v0, ov::op::v0::Xor); // Not marked as deprecated yet, but removed from new opsets

// _OPENVINO_OP_REG(BatchNormInference_v0, ov::op::v0::BatchNormInference);
// _OPENVINO_OP_REG(Range_v0, ov::op::v0::Range);
// _OPENVINO_OP_REG(RNNCell_v0, ov::op::v0::RNNCell);

// ------------------------------ Supported v1 ops ------------------------------ //
_OPENVINO_OP_REG(Add_v1, ov::op::v1::Add);
_OPENVINO_OP_REG(AvgPool_v1, ov::op::v1::AvgPool);
_OPENVINO_OP_REG(BatchToSpace_v1, ov::op::v1::BatchToSpace);
// _OPENVINO_OP_REG(BinaryConvolution_v1, ov::op::v1::BinaryConvolution); Supported via BinaryConvolution->Convolution conversion
_OPENVINO_OP_REG(Broadcast_v1, ov::op::v1::Broadcast);
_OPENVINO_OP_REG(ConvertLike_v1, ov::op::v1::ConvertLike);
_OPENVINO_OP_REG(Convolution_v1, ov::op::v1::Convolution);
_OPENVINO_OP_REG(ConvolutionBackpropData_v1, ov::op::v1::ConvolutionBackpropData);
_OPENVINO_OP_REG(DeformableConvolution_v1, ov::op::v1::DeformableConvolution);
_OPENVINO_OP_REG(DeformablePSROIPooling_v1, ov::op::v1::DeformablePSROIPooling);
_OPENVINO_OP_REG(Divide_v1, ov::op::v1::Divide);
_OPENVINO_OP_REG(Equal_v1, ov::op::v1::Equal);
_OPENVINO_OP_REG(FloorMod_v1, ov::op::v1::FloorMod);
_OPENVINO_OP_REG(Gather_v1, ov::op::v1::Gather);
_OPENVINO_OP_REG(GatherTree_v1, ov::op::v1::GatherTree);
_OPENVINO_OP_REG(Greater_v1, ov::op::v1::Greater);
_OPENVINO_OP_REG(GreaterEqual_v1, ov::op::v1::GreaterEqual);
_OPENVINO_OP_REG(GroupConvolution_v1, ov::op::v1::GroupConvolution);
_OPENVINO_OP_REG(GroupConvolutionBackpropData_v1, ov::op::v1::GroupConvolutionBackpropData);
_OPENVINO_OP_REG(Less_v1, ov::op::v1::Less);
_OPENVINO_OP_REG(LessEqual_v1, ov::op::v1::LessEqual);
_OPENVINO_OP_REG(LogicalAnd_v1, ov::op::v1::LogicalAnd);
_OPENVINO_OP_REG(LogicalNot_v1, ov::op::v1::LogicalNot);
_OPENVINO_OP_REG(LogicalOr_v1, ov::op::v1::LogicalOr);
_OPENVINO_OP_REG(LogicalXor_v1, ov::op::v1::LogicalXor);
_OPENVINO_OP_REG(MaxPool_v1, ov::op::v1::MaxPool);
_OPENVINO_OP_REG(Maximum_v1, ov::op::v1::Maximum);
_OPENVINO_OP_REG(Minimum_v1, ov::op::v1::Minimum);
_OPENVINO_OP_REG(Multiply_v1, ov::op::v1::Multiply);
_OPENVINO_OP_REG(NotEqual_v1, ov::op::v1::NotEqual);
// _OPENVINO_OP_REG(NonMaxSuppression_v1, ov::op::v1::NonMaxSuppression); Supported via v1 -> v5 internal conversion
_OPENVINO_OP_REG(OneHot_v1, ov::op::v1::OneHot);
_OPENVINO_OP_REG(Pad_v1, ov::op::v1::Pad);
_OPENVINO_OP_REG(Power_v1, ov::op::v1::Power);
_OPENVINO_OP_REG(ReduceMax_v1, ov::op::v1::ReduceMax);
_OPENVINO_OP_REG(ReduceLogicalAnd_v1, ov::op::v1::ReduceLogicalAnd);
_OPENVINO_OP_REG(ReduceLogicalOr_v1, ov::op::v1::ReduceLogicalOr);
_OPENVINO_OP_REG(ReduceMean_v1, ov::op::v1::ReduceMean);
_OPENVINO_OP_REG(ReduceMin_v1, ov::op::v1::ReduceMin);
_OPENVINO_OP_REG(ReduceProd_v1, ov::op::v1::ReduceProd);
_OPENVINO_OP_REG(ReduceSum_v1, ov::op::v1::ReduceSum);
_OPENVINO_OP_REG(Reshape_v1, ov::op::v1::Reshape);
_OPENVINO_OP_REG(Reverse_v1, ov::op::v1::Reverse);
_OPENVINO_OP_REG(Subtract_v1, ov::op::v1::Subtract);
_OPENVINO_OP_REG(SpaceToBatch_v1, ov::op::v1::SpaceToBatch);
_OPENVINO_OP_REG(Softmax_v1, ov::op::v1::Softmax);
_OPENVINO_OP_REG(StridedSlice_v1, ov::op::v1::StridedSlice);
_OPENVINO_OP_REG(Select_v1, ov::op::v1::Select);
_OPENVINO_OP_REG(Split_v1, ov::op::v1::Split);
_OPENVINO_OP_REG(Transpose_v1, ov::op::v1::Transpose);
_OPENVINO_OP_REG(TopK_v1, ov::op::v1::TopK);
_OPENVINO_OP_REG(VariadicSplit_v1, ov::op::v1::VariadicSplit);
_OPENVINO_OP_REG(Mod_v1, ov::op::v1::Mod);

// ------------------------------ Supported v3 ops ------------------------------ //
_OPENVINO_OP_REG(Asinh_v3, ov::op::v3::Asinh);
_OPENVINO_OP_REG(Acosh_v3, ov::op::v3::Acosh);
_OPENVINO_OP_REG(Atanh_v3, ov::op::v3::Atanh);
_OPENVINO_OP_REG(Broadcast_v3, ov::op::v3::Broadcast);
_OPENVINO_OP_REG(Bucketize_v3, ov::op::v3::Bucketize);
_OPENVINO_OP_REG(EmbeddingBagOffsetsSum_v3, ov::op::v3::EmbeddingBagOffsetsSum);
_OPENVINO_OP_REG(EmbeddingBagPackedSum_v3, ov::op::v3::EmbeddingBagPackedSum);
_OPENVINO_OP_REG(EmbeddingSegmentsSum_v3, ov::op::v3::EmbeddingSegmentsSum);
_OPENVINO_OP_REG(ExtractImagePatches_v3, ov::op::v3::ExtractImagePatches);
_OPENVINO_OP_REG(NonZero_v3, ov::op::v3::NonZero);
_OPENVINO_OP_REG(ROIAlign_v3, ov::op::v3::ROIAlign);
_OPENVINO_OP_REG(ScatterUpdate_v3, ov::op::v3::ScatterUpdate);
_OPENVINO_OP_REG(ScatterElementsUpdate_v3, ov::op::v3::ScatterElementsUpdate);
_OPENVINO_OP_REG(ScatterNDUpdate_v3, ov::op::v3::ScatterNDUpdate);
_OPENVINO_OP_REG(ShapeOf_v3, ov::op::v3::ShapeOf);
_OPENVINO_OP_REG(Assign_v3, ov::op::v3::Assign);
_OPENVINO_OP_REG(ReadValue_v3, ov::op::v3::ReadValue);
// _OPENVINO_OP_REG(NonMaxSuppression_v3, ov::op::v3::NonMaxSuppression); Supported via v3 -> v5 internal conversion

// ----------------------------- Unsupported v3 ops ----------------------------- //
// _OPENVINO_OP_REG(GRUCell_v3, ov::op::v3::GRUCell);
// _OPENVINO_OP_REG(NonZero_v3, ov::op::v3::NonZero);
// _OPENVINO_OP_REG(TopK_v3, ov::op::v3::TopK);

// ------------------------------ Supported v4 ops ------------------------------ //
_OPENVINO_OP_REG(HSwish_v4, ov::op::v4::HSwish);
_OPENVINO_OP_REG(Interpolate_v4, ov::op::v4::Interpolate);
_OPENVINO_OP_REG(LSTMCell_v4, ov::op::v4::LSTMCell);
_OPENVINO_OP_REG(Mish_v4, ov::op::v4::Mish);
// _OPENVINO_OP_REG(NonMaxSuppression_v4, ov::op::v4::NonMaxSuppression); Supported via v4 -> v5 internal conversion
_OPENVINO_OP_REG(Proposal_v4, ov::op::v4::Proposal);
_OPENVINO_OP_REG(Range_v4, ov::op::v4::Range);
_OPENVINO_OP_REG(ReduceL1_v4, ov::op::v4::ReduceL1);
_OPENVINO_OP_REG(ReduceL2_v4, ov::op::v4::ReduceL2);
_OPENVINO_OP_REG(SoftPlus_v4, ov::op::v4::SoftPlus);
_OPENVINO_OP_REG(Swish_v4, ov::op::v4::Swish);
_OPENVINO_OP_REG(CTCLoss_v4, ov::op::v4::CTCLoss);

// ----------------------------- Unsupported v4 ops ----------------------------- //
// _OPENVINO_OP_REG(Range_v4, ov::op::v4::Range);

// ------------------------------ Supported v5 ops ------------------------------ //
_OPENVINO_OP_REG(HSigmoid_v5, ov::op::v5::HSigmoid);
_OPENVINO_OP_REG(LogSoftmax_v5, ov::op::v5::LogSoftmax);
_OPENVINO_OP_REG(LSTMSequence_v5, ov::op::v5::LSTMSequence);
// _OPENVINO_OP_REG(NonMaxSuppression_v5, ov::op::v5::NonMaxSuppression); Supported via v5 -> v5 internal conversion
_OPENVINO_OP_REG(Round_v5, ov::op::v5::Round);
_OPENVINO_OP_REG(GatherND_v5, ov::op::v5::GatherND);
_OPENVINO_OP_REG(Loop_v5, ov::op::v5::Loop);

// ----------------------------- Unsupported v5 ops ----------------------------- //
// _OPENVINO_OP_REG(BatchNormInference_v5, ov::op::v5::BatchNormInference);
// _OPENVINO_OP_REG(GRUSequence_v5, ov::op::v5::GRUSequence);
// _OPENVINO_OP_REG(RNNSequence_v5, ov::op::v5::RNNSequence);

// ------------------------------ Supported v6 ops ------------------------------ //
_OPENVINO_OP_REG(CTCGreedyDecoderSeqLen_v6, ov::op::v6::CTCGreedyDecoderSeqLen);
_OPENVINO_OP_REG(MVN_v6, ov::op::v6::MVN);
_OPENVINO_OP_REG(GatherElements_v6, ov::op::v6::GatherElements);
_OPENVINO_OP_REG(ExperimentalDetectronPriorGridGenerator_v6, ov::op::v6::ExperimentalDetectronPriorGridGenerator);
_OPENVINO_OP_REG(ExperimentalDetectronROIFeatureExtractor_v6, ov::op::v6::ExperimentalDetectronROIFeatureExtractor);
_OPENVINO_OP_REG(ExperimentalDetectronTopKROIs_v6, ov::op::v6::ExperimentalDetectronTopKROIs)
_OPENVINO_OP_REG(ExperimentalDetectronGenerateProposalsSingleImage_v6, ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage);
_OPENVINO_OP_REG(ExperimentalDetectronDetectionOutput_v6, ov::op::v6::ExperimentalDetectronDetectionOutput);
_OPENVINO_OP_REG(Assign_v6, ov::op::v6::Assign);
_OPENVINO_OP_REG(ReadValue_v6, ov::op::v6::ReadValue);

// ------------------------------ Supported v7 ops ------------------------------ //
_OPENVINO_OP_REG(DFT_v7, ov::op::v7::DFT);
_OPENVINO_OP_REG(Gather_v7, ov::op::v7::Gather);
_OPENVINO_OP_REG(Gelu_v7, ov::op::v7::Gelu);
_OPENVINO_OP_REG(IDFT_v7, ov::op::v7::IDFT);
_OPENVINO_OP_REG(Roll_v7, ov::op::v7::Roll);

// ------------------------------ Supported v8 ops ------------------------------ //
_OPENVINO_OP_REG(Slice_v8, ov::op::v8::Slice);
_OPENVINO_OP_REG(Gather_v8, ov::op::v8::Gather);
_OPENVINO_OP_REG(GatherND_v8, ov::op::v8::GatherND);
_OPENVINO_OP_REG(DetectionOutput_v8, ov::op::v8::DetectionOutput);
_OPENVINO_OP_REG(DeformableConvolution_v8, ov::op::v8::DeformableConvolution);
_OPENVINO_OP_REG(NV12toRGB_v8, ov::op::v8::NV12toRGB);
_OPENVINO_OP_REG(NV12toBGR_v8, ov::op::v8::NV12toBGR);
_OPENVINO_OP_REG(I420toRGB_v8, ov::op::v8::I420toRGB);
_OPENVINO_OP_REG(I420toBGR_v8, ov::op::v8::I420toBGR);
// _OPENVINO_OP_REG(RandomUniform_v8, ov::op::v8::RandomUniform)
_OPENVINO_OP_REG(MaxPool_v8, ov::op::v8::MaxPool);
_OPENVINO_OP_REG(AdaptiveAvgPool_v8, ov::op::v8::AdaptiveAvgPool);
_OPENVINO_OP_REG(AdaptiveMaxPool_v8, ov::op::v8::AdaptiveMaxPool);
_OPENVINO_OP_REG(Softmax_v8, ov::op::v8::Softmax);
_OPENVINO_OP_REG(PriorBox_v8, ov::op::v8::PriorBox);
// _OPENVINO_OP_REG(If_v8, ov::op::v8::If);

// ------------------------------ Supported v9 ops ------------------------------ //
_OPENVINO_OP_REG(GridSample_v9, ov::op::v9::GridSample)
_OPENVINO_OP_REG(SoftSign_v9, ov::op::v9::SoftSign)
_OPENVINO_OP_REG(ROIAlign_v9, ov::op::v9::ROIAlign);
_OPENVINO_OP_REG(RDFT_v9, ov::op::v9::RDFT);
_OPENVINO_OP_REG(IRDFT_v9, ov::op::v9::IRDFT);
_OPENVINO_OP_REG(Eye_v9, ov::op::v9::Eye);

// ------------------------------ Supported v10 ops ----------------------------- //
_OPENVINO_OP_REG(IsFinite_v10, ov::op::v10::IsFinite);
_OPENVINO_OP_REG(IsInf_v10, ov::op::v10::IsInf);
_OPENVINO_OP_REG(IsNaN_v10, ov::op::v10::IsNaN);
_OPENVINO_OP_REG(Unique_v10, ov::op::v10::Unique);

// ------------------------------ Supported v11 ops ----------------------------- //
_OPENVINO_OP_REG(Interpolate_v11, ov::op::v11::Interpolate);
_OPENVINO_OP_REG(TopK_v11, ov::op::v11::TopK);

// ------------------------------ Supported v12 ops ----------------------------- //
_OPENVINO_OP_REG(GroupNormalization_v12, ov::op::v12::GroupNormalization);
_OPENVINO_OP_REG(Pad_v12, ov::op::v12::Pad);
_OPENVINO_OP_REG(ScatterElementsUpdate_v12, ov::op::v12::ScatterElementsUpdate);

// ------------------------------ Supported v13 ops ----------------------------- //
_OPENVINO_OP_REG(Multinomial_v13, ov::op::v13::Multinomial);

// --------------------------- Supported internal ops --------------------------- //
// _OPENVINO_OP_REG(NonMaxSuppressionIEInternal, ov::op::internal::NonMaxSuppressionIEInternal);
// _OPENVINO_OP_REG(GenerateProposalsIEInternal, ov::op::internal::GenerateProposalsIEInternal);
// _OPENVINO_OP_REG(NmsStaticShapeIE8, ov::op::internal::NmsStaticShapeIE8);
// _OPENVINO_OP_REG(MulticlassNmsIEInternal, ov::op::internal::MulticlassNmsIEInternal);
_OPENVINO_OP_REG(FullyConnected, ov::intel_gpu::op::FullyConnected);
_OPENVINO_OP_REG(FullyConnectedCompressed, ov::intel_gpu::op::FullyConnectedCompressed);
_OPENVINO_OP_REG(RMS, ov::intel_gpu::op::RMS);
_OPENVINO_OP_REG(Reorder, ov::intel_gpu::op::Reorder);
_OPENVINO_OP_REG(GatherCompressed_internal, ov::intel_gpu::op::GatherCompressed);
_OPENVINO_OP_REG(KVCache_internal, ov::intel_gpu::op::KVCache);
_OPENVINO_OP_REG(ReadValue_internal, ov::intel_gpu::op::ReadValue);
_OPENVINO_OP_REG(Placeholder_internal, ov::intel_gpu::op::Placeholder);
_OPENVINO_OP_REG(Convolution_internal, ov::intel_gpu::op::Convolution);
