// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef REGISTER_FACTORY
#    error "REGISTER_FACTORY is not defined"
#endif

// ------------------------------ Supported v0 ops ------------------------------ //
// REGISTER_FACTORY(Abs_v0, ov::op::v0::Abs);
REGISTER_FACTORY(Acos_v0, ov::op::v0::Acos);
REGISTER_FACTORY(Asin_v0, ov::op::v0::Asin);
REGISTER_FACTORY(Atan_v0, ov::op::v0::Atan);
REGISTER_FACTORY(Ceiling_v0, ov::op::v0::Ceiling);
REGISTER_FACTORY(Clamp_v0, ov::op::v0::Clamp);
REGISTER_FACTORY(Concat_v0, ov::op::v0::Concat);
REGISTER_FACTORY(Constant_v0, ov::op::v0::Constant);
REGISTER_FACTORY(Convert_v0, ov::op::v0::Convert);
REGISTER_FACTORY(Cos_v0, ov::op::v0::Cos);
REGISTER_FACTORY(Cosh_v0, ov::op::v0::Cosh);
REGISTER_FACTORY(CumSum_v0, ov::op::v0::CumSum);
REGISTER_FACTORY(CTCGreedyDecoder_v0, ov::op::v0::CTCGreedyDecoder);
REGISTER_FACTORY(DepthToSpace_v0, ov::op::v0::DepthToSpace);
REGISTER_FACTORY(DetectionOutput_v0, ov::op::v0::DetectionOutput);
REGISTER_FACTORY(Elu_v0, ov::op::v0::Elu);
REGISTER_FACTORY(Erf_v0, ov::op::v0::Erf);
REGISTER_FACTORY(Exp_v0, ov::op::v0::Exp);
REGISTER_FACTORY(FakeQuantize_v0, ov::op::v0::FakeQuantize);
REGISTER_FACTORY(Floor_v0, ov::op::v0::Floor);
REGISTER_FACTORY(Gelu_v0, ov::op::v0::Gelu);
REGISTER_FACTORY(GRN_v0, ov::op::v0::GRN);
REGISTER_FACTORY(HardSigmoid_v0, ov::op::v0::HardSigmoid);
// REGISTER_FACTORY(Interpolate_v0, ov::op::v0::Interpolate); Supported via v0 -> v4 conversion
REGISTER_FACTORY(Log_v0, ov::op::v0::Log);
REGISTER_FACTORY(LRN_v0, ov::op::v0::LRN);
REGISTER_FACTORY(MatMul_v0, ov::op::v0::MatMul);
REGISTER_FACTORY(MVN_v0, ov::op::v0::MVN);
REGISTER_FACTORY(Negative_v0, ov::op::v0::Negative);
REGISTER_FACTORY(NormalizeL2_v0, ov::op::v0::NormalizeL2);
REGISTER_FACTORY(Parameter_v0, ov::op::v0::Parameter);
REGISTER_FACTORY(PRelu_v0, ov::op::v0::PRelu);
REGISTER_FACTORY(PriorBox_v0, ov::op::v0::PriorBox);
REGISTER_FACTORY(PriorBoxClustered_v0, ov::op::v0::PriorBoxClustered);
REGISTER_FACTORY(Proposal_v0, ov::op::v0::Proposal);
REGISTER_FACTORY(PSROIPooling_v0, ov::op::v0::PSROIPooling);
// REGISTER_FACTORY(Relu_v0, ov::op::v0::Relu);
REGISTER_FACTORY(Result_v0, ov::op::v0::Result);
REGISTER_FACTORY(RegionYolo_v0, ov::op::v0::RegionYolo);
REGISTER_FACTORY(ReorgYolo_v0, ov::op::v0::ReorgYolo);
REGISTER_FACTORY(ReverseSequence_v0, ov::op::v0::ReverseSequence);
REGISTER_FACTORY(ROIPooling_v0, ov::op::v0::ROIPooling);
REGISTER_FACTORY(Sigmoid_v0, ov::op::v0::Sigmoid);
REGISTER_FACTORY(Sqrt_v0, ov::op::v0::Sqrt);
REGISTER_FACTORY(Selu_v0, ov::op::v0::Selu);
REGISTER_FACTORY(Sin_v0, ov::op::v0::Sin);
REGISTER_FACTORY(Sinh_v0, ov::op::v0::Sinh);
REGISTER_FACTORY(Sign_v0, ov::op::v0::Sign);
REGISTER_FACTORY(SquaredDifference_v0, ov::op::v0::SquaredDifference);
REGISTER_FACTORY(SpaceToDepth_v0, ov::op::v0::SpaceToDepth);
REGISTER_FACTORY(Squeeze_v0, ov::op::v0::Squeeze);
REGISTER_FACTORY(ShapeOf_v0, ov::op::v0::ShapeOf);
REGISTER_FACTORY(ShuffleChannels_v0, ov::op::v0::ShuffleChannels);
REGISTER_FACTORY(Tan_v0, ov::op::v0::Tan);
REGISTER_FACTORY(Tanh_v0, ov::op::v0::Tanh);
// REGISTER_FACTORY(TensorIterator_v0, ov::op::v0::TensorIterator);
REGISTER_FACTORY(Tile_v0, ov::op::v0::Tile);
REGISTER_FACTORY(Unsqueeze_v0, ov::op::v0::Unsqueeze);

// ----------------------------- Unsupported v0 ops ----------------------------- //
// Deprecated ops
// REGISTER_FACTORY(Add_v0, ov::op::v0::Add);
// REGISTER_FACTORY(Divide_v0, ov::op::v0::Divide);
// REGISTER_FACTORY(Greater_v0, ov::op::v0::Greater);
// REGISTER_FACTORY(GreaterEq_v0, ov::op::v0::GreaterEq);
// REGISTER_FACTORY(Less_v0, ov::op::v0::Less);
// REGISTER_FACTORY(LessEq_v0, ov::op::v0::LessEq);
// REGISTER_FACTORY(LSTMSequence_v0, ov::op::v0::LSTMSequence);
// REGISTER_FACTORY(LSTMCell_v0, ov::op::v0::LSTMCell);
// REGISTER_FACTORY(Maximum_v0, ov::op::v0::Maximum);
// REGISTER_FACTORY(Minimum_v0, ov::op::v0::Minimum);
// REGISTER_FACTORY(Multiply_v0, ov::op::v0::Multiply);
// REGISTER_FACTORY(NotEqual_v0, ov::op::v0::NotEqual);
// REGISTER_FACTORY(Power_v0, ov::op::v0::Power);
// REGISTER_FACTORY(Quantize_v0, ov::op::v0::Quantize);
// REGISTER_FACTORY(Select_v0, ov::op::v0::Select);
// REGISTER_FACTORY(Subtract_v0, ov::op::v0::Subtract);
// REGISTER_FACTORY(Xor_v0, ov::op::v0::Xor); // Not marked as deprecated yet, but removed from new opsets

// REGISTER_FACTORY(BatchNormInference_v0, ov::op::v0::BatchNormInference);
// REGISTER_FACTORY(Range_v0, ov::op::v0::Range);
// REGISTER_FACTORY(RNNCell_v0, ov::op::v0::RNNCell);

// ------------------------------ Supported v1 ops ------------------------------ //
REGISTER_FACTORY(Add_v1, ov::op::v1::Add);
REGISTER_FACTORY(AvgPool_v1, ov::op::v1::AvgPool);
// REGISTER_FACTORY(BatchToSpace_v1, ov::op::v1::BatchToSpace);
// REGISTER_FACTORY(BinaryConvolution_v1, ov::op::v1::BinaryConvolution); Supported via BinaryConvolution->Convolution conversion
REGISTER_FACTORY(Broadcast_v1, ov::op::v1::Broadcast);
REGISTER_FACTORY(ConvertLike_v1, ov::op::v1::ConvertLike);
// REGISTER_FACTORY(Convolution_v1, ov::op::v1::Convolution);
REGISTER_FACTORY(ConvolutionBackpropData_v1, ov::op::v1::ConvolutionBackpropData);
REGISTER_FACTORY(DeformableConvolution_v1, ov::op::v1::DeformableConvolution);
REGISTER_FACTORY(DeformablePSROIPooling_v1, ov::op::v1::DeformablePSROIPooling);
REGISTER_FACTORY(Divide_v1, ov::op::v1::Divide);
REGISTER_FACTORY(Equal_v1, ov::op::v1::Equal);
REGISTER_FACTORY(FloorMod_v1, ov::op::v1::FloorMod);
REGISTER_FACTORY(Gather_v1, ov::op::v1::Gather);
REGISTER_FACTORY(GatherTree_v1, ov::op::v1::GatherTree);
REGISTER_FACTORY(Greater_v1, ov::op::v1::Greater);
REGISTER_FACTORY(GreaterEqual_v1, ov::op::v1::GreaterEqual);
// REGISTER_FACTORY(GroupConvolution_v1, ov::op::v1::GroupConvolution);
REGISTER_FACTORY(GroupConvolutionBackpropData_v1, ov::op::v1::GroupConvolutionBackpropData);
REGISTER_FACTORY(Less_v1, ov::op::v1::Less);
REGISTER_FACTORY(LessEqual_v1, ov::op::v1::LessEqual);
REGISTER_FACTORY(LogicalAnd_v1, ov::op::v1::LogicalAnd);
REGISTER_FACTORY(LogicalNot_v1, ov::op::v1::LogicalNot);
REGISTER_FACTORY(LogicalOr_v1, ov::op::v1::LogicalOr);
REGISTER_FACTORY(LogicalXor_v1, ov::op::v1::LogicalXor);
REGISTER_FACTORY(MaxPool_v1, ov::op::v1::MaxPool);
REGISTER_FACTORY(Maximum_v1, ov::op::v1::Maximum);
REGISTER_FACTORY(Minimum_v1, ov::op::v1::Minimum);
REGISTER_FACTORY(Multiply_v1, ov::op::v1::Multiply);
REGISTER_FACTORY(NotEqual_v1, ov::op::v1::NotEqual);
// REGISTER_FACTORY(NonMaxSuppression_v1, ov::op::v1::NonMaxSuppression); Supported via v1 -> v5 internal conversion
REGISTER_FACTORY(OneHot_v1, ov::op::v1::OneHot);
REGISTER_FACTORY(Pad_v1, ov::op::v1::Pad);
REGISTER_FACTORY(Power_v1, ov::op::v1::Power);
REGISTER_FACTORY(ReduceMax_v1, ov::op::v1::ReduceMax);
REGISTER_FACTORY(ReduceLogicalAnd_v1, ov::op::v1::ReduceLogicalAnd);
REGISTER_FACTORY(ReduceLogicalOr_v1, ov::op::v1::ReduceLogicalOr);
REGISTER_FACTORY(ReduceMean_v1, ov::op::v1::ReduceMean);
REGISTER_FACTORY(ReduceMin_v1, ov::op::v1::ReduceMin);
REGISTER_FACTORY(ReduceProd_v1, ov::op::v1::ReduceProd);
REGISTER_FACTORY(ReduceSum_v1, ov::op::v1::ReduceSum);
REGISTER_FACTORY(Reshape_v1, ov::op::v1::Reshape);
REGISTER_FACTORY(Reverse_v1, ov::op::v1::Reverse);
REGISTER_FACTORY(Subtract_v1, ov::op::v1::Subtract);
REGISTER_FACTORY(SpaceToBatch_v1, ov::op::v1::SpaceToBatch);
REGISTER_FACTORY(Softmax_v1, ov::op::v1::Softmax);
REGISTER_FACTORY(StridedSlice_v1, ov::op::v1::StridedSlice);
REGISTER_FACTORY(Select_v1, ov::op::v1::Select);
REGISTER_FACTORY(Split_v1, ov::op::v1::Split);
REGISTER_FACTORY(Transpose_v1, ov::op::v1::Transpose);
REGISTER_FACTORY(TopK_v1, ov::op::v1::TopK);
REGISTER_FACTORY(VariadicSplit_v1, ov::op::v1::VariadicSplit);
REGISTER_FACTORY(Mod_v1, ov::op::v1::Mod);

// ------------------------------ Supported v3 ops ------------------------------ //
REGISTER_FACTORY(Asinh_v3, ov::op::v3::Asinh);
REGISTER_FACTORY(Acosh_v3, ov::op::v3::Acosh);
REGISTER_FACTORY(Atanh_v3, ov::op::v3::Atanh);
REGISTER_FACTORY(Broadcast_v3, ov::op::v3::Broadcast);
REGISTER_FACTORY(Bucketize_v3, ov::op::v3::Bucketize);
REGISTER_FACTORY(EmbeddingBagOffsetsSum_v3, ov::op::v3::EmbeddingBagOffsetsSum);
REGISTER_FACTORY(EmbeddingBagPackedSum_v3, ov::op::v3::EmbeddingBagPackedSum);
REGISTER_FACTORY(EmbeddingSegmentsSum_v3, ov::op::v3::EmbeddingSegmentsSum);
REGISTER_FACTORY(ExtractImagePatches_v3, ov::op::v3::ExtractImagePatches);
REGISTER_FACTORY(NonZero_v3, ov::op::v3::NonZero);
REGISTER_FACTORY(ROIAlign_v3, ov::op::v3::ROIAlign);
REGISTER_FACTORY(ScatterUpdate_v3, ov::op::v3::ScatterUpdate);
REGISTER_FACTORY(ScatterElementsUpdate_v3, ov::op::v3::ScatterElementsUpdate);
REGISTER_FACTORY(ScatterNDUpdate_v3, ov::op::v3::ScatterNDUpdate);
REGISTER_FACTORY(ShapeOf_v3, ov::op::v3::ShapeOf);
REGISTER_FACTORY(Assign_v3, ov::op::v3::Assign);
REGISTER_FACTORY(ReadValue_v3, ov::op::v3::ReadValue);
// REGISTER_FACTORY(NonMaxSuppression_v3, ov::op::v3::NonMaxSuppression); Supported via v3 -> v5 internal conversion

// ----------------------------- Unsupported v3 ops ----------------------------- //
// REGISTER_FACTORY(GRUCell_v3, ov::op::v3::GRUCell);
// REGISTER_FACTORY(NonZero_v3, ov::op::v3::NonZero);
// REGISTER_FACTORY(TopK_v3, ov::op::v3::TopK);

// ------------------------------ Supported v4 ops ------------------------------ //
REGISTER_FACTORY(HSwish_v4, ov::op::v4::HSwish);
REGISTER_FACTORY(Interpolate_v4, ov::op::v4::Interpolate);
REGISTER_FACTORY(LSTMCell_v4, ov::op::v4::LSTMCell);
REGISTER_FACTORY(Mish_v4, ov::op::v4::Mish);
// REGISTER_FACTORY(NonMaxSuppression_v4, ov::op::v4::NonMaxSuppression); Supported via v4 -> v5 internal conversion
REGISTER_FACTORY(Proposal_v4, ov::op::v4::Proposal);
REGISTER_FACTORY(Range_v4, ov::op::v4::Range);
REGISTER_FACTORY(ReduceL1_v4, ov::op::v4::ReduceL1);
REGISTER_FACTORY(ReduceL2_v4, ov::op::v4::ReduceL2);
REGISTER_FACTORY(SoftPlus_v4, ov::op::v4::SoftPlus);
REGISTER_FACTORY(Swish_v4, ov::op::v4::Swish);
REGISTER_FACTORY(CTCLoss_v4, ov::op::v4::CTCLoss);

// ----------------------------- Unsupported v4 ops ----------------------------- //
// REGISTER_FACTORY(Range_v4, ov::op::v4::Range);

// ------------------------------ Supported v5 ops ------------------------------ //
REGISTER_FACTORY(HSigmoid_v5, ov::op::v5::HSigmoid);
REGISTER_FACTORY(LogSoftmax_v5, ov::op::v5::LogSoftmax);
REGISTER_FACTORY(LSTMSequence_v5, ov::op::v5::LSTMSequence);
// REGISTER_FACTORY(NonMaxSuppression_v5, ov::op::v5::NonMaxSuppression); Supported via v5 -> v5 internal conversion
REGISTER_FACTORY(Round_v5, ov::op::v5::Round);
REGISTER_FACTORY(GatherND_v5, ov::op::v5::GatherND);
REGISTER_FACTORY(Loop_v5, ov::op::v5::Loop);

// ----------------------------- Unsupported v5 ops ----------------------------- //
// REGISTER_FACTORY(BatchNormInference_v5, ov::op::v5::BatchNormInference);
// REGISTER_FACTORY(GRUSequence_v5, ov::op::v5::GRUSequence);
// REGISTER_FACTORY(RNNSequence_v5, ov::op::v5::RNNSequence);

// ------------------------------ Supported v6 ops ------------------------------ //
REGISTER_FACTORY(CTCGreedyDecoderSeqLen_v6, ov::op::v6::CTCGreedyDecoderSeqLen);
REGISTER_FACTORY(MVN_v6, ov::op::v6::MVN);
REGISTER_FACTORY(GatherElements_v6, ov::op::v6::GatherElements);
REGISTER_FACTORY(ExperimentalDetectronPriorGridGenerator_v6, ov::op::v6::ExperimentalDetectronPriorGridGenerator);
REGISTER_FACTORY(ExperimentalDetectronROIFeatureExtractor_v6, ov::op::v6::ExperimentalDetectronROIFeatureExtractor);
REGISTER_FACTORY(ExperimentalDetectronTopKROIs_v6, ov::op::v6::ExperimentalDetectronTopKROIs)
REGISTER_FACTORY(ExperimentalDetectronGenerateProposalsSingleImage_v6, ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage);
REGISTER_FACTORY(ExperimentalDetectronDetectionOutput_v6, ov::op::v6::ExperimentalDetectronDetectionOutput);
REGISTER_FACTORY(Assign_v6, ov::op::v6::Assign);
REGISTER_FACTORY(ReadValue_v6, ov::op::v6::ReadValue);

// ------------------------------ Supported v7 ops ------------------------------ //
REGISTER_FACTORY(DFT_v7, ov::op::v7::DFT);
REGISTER_FACTORY(Gather_v7, ov::op::v7::Gather);
REGISTER_FACTORY(Gelu_v7, ov::op::v7::Gelu);
REGISTER_FACTORY(IDFT_v7, ov::op::v7::IDFT);
REGISTER_FACTORY(Roll_v7, ov::op::v7::Roll);

// ------------------------------ Supported v8 ops ------------------------------ //
REGISTER_FACTORY(Slice_v8, ov::op::v8::Slice);
REGISTER_FACTORY(Gather_v8, ov::op::v8::Gather);
REGISTER_FACTORY(GatherND_v8, ov::op::v8::GatherND);
REGISTER_FACTORY(DetectionOutput_v8, ov::op::v8::DetectionOutput);
REGISTER_FACTORY(DeformableConvolution_v8, ov::op::v8::DeformableConvolution);
REGISTER_FACTORY(NV12toRGB_v8, ov::op::v8::NV12toRGB);
REGISTER_FACTORY(NV12toBGR_v8, ov::op::v8::NV12toBGR);
REGISTER_FACTORY(I420toRGB_v8, ov::op::v8::I420toRGB);
REGISTER_FACTORY(I420toBGR_v8, ov::op::v8::I420toBGR);
// REGISTER_FACTORY(RandomUniform_v8, ov::op::v8::RandomUniform)
REGISTER_FACTORY(MaxPool_v8, ov::op::v8::MaxPool);
REGISTER_FACTORY(AdaptiveAvgPool_v8, ov::op::v8::AdaptiveAvgPool);
REGISTER_FACTORY(AdaptiveMaxPool_v8, ov::op::v8::AdaptiveMaxPool);
REGISTER_FACTORY(Softmax_v8, ov::op::v8::Softmax);
REGISTER_FACTORY(PriorBox_v8, ov::op::v8::PriorBox);
// REGISTER_FACTORY(If_v8, ov::op::v8::If);

// ------------------------------ Supported v9 ops ------------------------------ //
REGISTER_FACTORY(GridSample_v9, ov::op::v9::GridSample)
REGISTER_FACTORY(SoftSign_v9, ov::op::v9::SoftSign)
REGISTER_FACTORY(ROIAlign_v9, ov::op::v9::ROIAlign);
REGISTER_FACTORY(RDFT_v9, ov::op::v9::RDFT);
REGISTER_FACTORY(IRDFT_v9, ov::op::v9::IRDFT);
REGISTER_FACTORY(Eye_v9, ov::op::v9::Eye);

// ------------------------------ Supported v10 ops ----------------------------- //
REGISTER_FACTORY(IsFinite_v10, ov::op::v10::IsFinite);
REGISTER_FACTORY(IsInf_v10, ov::op::v10::IsInf);
REGISTER_FACTORY(IsNaN_v10, ov::op::v10::IsNaN);
REGISTER_FACTORY(Unique_v10, ov::op::v10::Unique);

// ------------------------------ Supported v11 ops ----------------------------- //
REGISTER_FACTORY(Interpolate_v11, ov::op::v11::Interpolate);
REGISTER_FACTORY(TopK_v11, ov::op::v11::TopK);

// ------------------------------ Supported v12 ops ----------------------------- //
REGISTER_FACTORY(GroupNormalization_v12, ov::op::v12::GroupNormalization);
REGISTER_FACTORY(Pad_v12, ov::op::v12::Pad);
REGISTER_FACTORY(ScatterElementsUpdate_v12, ov::op::v12::ScatterElementsUpdate);

// ------------------------------ Supported v13 ops ----------------------------- //
REGISTER_FACTORY(Multinomial_v13, ov::op::v13::Multinomial);

// --------------------------- Supported internal ops --------------------------- //
REGISTER_FACTORY(NonMaxSuppressionIEInternal_internal, ov::op::internal::NonMaxSuppressionIEInternal);
REGISTER_FACTORY(GenerateProposalsIEInternal_internal, ov::op::internal::GenerateProposalsIEInternal);
REGISTER_FACTORY(NmsStaticShapeIE8_internal, ov::op::internal::NmsStaticShapeIE<ov::op::v8::MatrixNms>);
REGISTER_FACTORY(MulticlassNmsIEInternal_internal, ov::op::internal::MulticlassNmsIEInternal);
REGISTER_FACTORY(FullyConnected_internal, ov::intel_gpu::op::FullyConnected);
// REGISTER_FACTORY(FullyConnectedCompressed_internal, ov::intel_gpu::op::FullyConnectedCompressed);
REGISTER_FACTORY(RMS_internal, ov::intel_gpu::op::RMS);
REGISTER_FACTORY(GatherCompressed_internal, ov::intel_gpu::op::GatherCompressed);
REGISTER_FACTORY(KVCache_internal, ov::intel_gpu::op::KVCache);
REGISTER_FACTORY(ReadValue_internal, ov::intel_gpu::op::ReadValue);
REGISTER_FACTORY(Gemm_internal, ov::intel_gpu::op::Gemm);
REGISTER_FACTORY(SwiGLU_internal, ov::intel_gpu::op::SwiGLU);
REGISTER_FACTORY(IndirectGemm_internal, ov::intel_gpu::op::IndirectGemm);
// REGISTER_FACTORY(Convolution_internal, ov::intel_gpu::op::Convolution);
REGISTER_FACTORY(Placeholder_internal, ov::intel_gpu::op::Placeholder);
