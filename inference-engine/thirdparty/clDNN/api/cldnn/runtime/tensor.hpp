// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compounds.hpp"
#include "utils.hpp"
#include "check.hpp"

#include <ngraph/partial_shape.hpp>

#include <map>
#include <list>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <vector>
#include <string>
#include <utility>
#include <functional>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_memory Memory description and management
/// @{

/// @brief Format information helper class.
struct format_traits {
    /// @brief Number of batch dimensions in a format.
    size_t batch_num;
    /// @brief Number of feature map/channel dimensions in a format.
    size_t feature_num;
    /// @brief Number of spatial (x,y) dimensions in a format.
    size_t spatial_num;
    /// @brief Number of local (x,y) dimensions in a format.
    size_t local_num;
    /// @brief Number of groups in a format.
    size_t group_num;
    /// @brief Dimensions changing order from rare to often.
    std::string order;
    /// @brief Dimensions order for internal storage.
    std::string internal_order;
    /// @brief Block sizes as a vector of pairs of dimension number and block size ordered from rare to often.
    std::vector<std::pair<size_t, int>> block_sizes;
    /// @brief Characters representing batch dimensions in an order.
    static const char* batch_chars() { return "bno"; }
    /// @brief Characters representing feature map/channel dimensions in an order.
    static const char* feature_chars() { return "fic"; }
    /// @brief Characters representing spatial dimensions in an order.
    static const char* spatial_chars() { return "xyzhsw"; }
    /// @brief Characters representing local dimensions in an order.
    static const char* local_chars() { return "kl"; }
    /// @brief Characters representing group dimensions in an order.
    static const char* group_chars() { return "g"; }
    /// @brief Checks if @p c represents batch dimension.
    static bool is_batch_char(char c) { return std::string(batch_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents feature map/channel dimension.
    static bool is_feature_char(char c) { return std::string(feature_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents spatial dimension.
    static bool is_spatial_char(char c) { return std::string(spatial_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents local dimensions.
    static bool is_local_char(char c) { return std::string(local_chars()).find_first_of(c) != std::string::npos; }
    /// @brief Checks if @p c represents group dimensions.
    static bool is_group_char(char c) { return std::string(group_chars()).find_first_of(c) != std::string::npos; }
};

/// @brief Represents memory formats (orders).
/// @n In CNN most of data is described as 4 dimensional blocks. In Intel(R) clDNN library we describe memory with 4 letters
/// - b - number of blocks in batch. For weights formats: output features - conv, neurons - inner product
/// - f - number of feature maps, features or channels. For weights formats: input features - conv, inputs, inner product
/// - x - spatial, width
/// - y - spatial, height
/// /n
/// For explanation how each format type is implemented in memory we will use naming shown bellow (b=2,f=3,y=3,x=3):
struct format {
    enum type : int32_t {
        // Data formats
        bfyx,                                   ///< the most common format for activations in clDNN.
        bfzyx,                                  ///< format for 5d data tensors
        bfwzyx,                                 ///< batch, feature, 4D spatial
        yxfb,                                   ///< batch first, feature and than spatials
        byxf,                                   ///< used in bitmaps, input from user i.e b images of RGB format
        fyxb,                                   ///< format not used inside clDNN, but supported in reorder as extension
                                                ///< for user provided formats.
        b_fs_yx_fsv16,                          ///< format used for blocked convolution
        b_fs_yx_fsv32,                          ///< format used for blocked int8 convolution
        b_fs_zyx_fsv16,                         ///< format used for 3D blocked convolution (features blocked by 16)
        b_fs_zyx_fsv32,                         ///< format used for blocked int8 3d convolution
        bs_fs_zyx_bsv16_fsv16,                  ///< format used for 3D blocked convolution (batch and features blocked by 16)
        bs_fs_yx_bsv16_fsv16,                   ///< format used for 2D blocked convolution (batch and features blocked by 16)
        fs_b_yx_fsv32,                          ///< format for input for fp16 primitives
        b_fs_yx_fsv4,                           ///< format for input for IMAD convolutions
        bs_xs_xsv8_bsv8,                        ///< format used only for fully connected weights: bs - batch slice,
                                                ///< xs - x slice, bsv8 - 8 values of single slice.
        bs_xs_xsv8_bsv16,                       ///< format used only for fully connected weights: bs - batch slice,
                                                ///< xs - x slice, bsv16 - 16 values of single slice.
        bs_x_bsv16,                             ///< format used only for fully connected weights fp16 batch=1 : bs - batch slice
                                                ///< (responses slice), bsv16 - 16 values of single batch slice, x - flattened plane of (fyx)
        b_fs_yx_32fp,                           ///< format for data for binary convolutions
        winograd_2x3_s1_data,                   ///< format used for input for winograd convolution, F(2,3) -- filter 3x3 with stride 1
        nv12,                                   ///< format for media nv12 input
        image_2d_rgba,                          ///< format for image2d RGBA, always allocates memory for 4 feature maps (even when only 3 are used)

        // Weights formats
        oiyx,                                         ///< the most common format for 2D weights
        ioyx,                                         ///< 2D weights format for deconvolutions
        yxio,                                         ///< format used 2D weights
        oizyx,                                        ///< the most common format for 3D convolution
        iozyx,                                        ///< 3D weights format for deconvolutions
        iyxo,
        os_iyx_osv16,                                 ///< format used only for convolution weights:
        os_is_yx_osv16_isv16,                         ///< format used for convolution i8 weights
        os_is_zyx_osv32_isv16,
        os_is_zyx_osv64_isv16,
        os_zyxi_osv16,                                ///< format used for weights for 3D convolution
        os_is_yx_isv16_osv16,                         ///< format used for blocked convolution
        os_is_zyx_isv16_osv16,                        ///< format used for weights for blocked 3D convolution
        is_os_zyx_isv16_osv16,                        ///< format used for weights for blocked 3D deconvolution
        is_os_yx_isv16_osv16,                         ///< format used for weights for blocked deconvolution
        os_is_yx_isv8_osv16_isv2,                     ///< format used for weights for blocked 2D convolution
        os_is_zyx_isv8_osv16_isv2,                    ///< format used for weights for blocked 3D convolution
                                                      ///< os - output feature maps slice, i - input feature maps,
                                                      ///< yx - spatials, sv16 - 16 values of single slice.
        os_iyx_osv32,                                 ///< format used only for convolution weights:
                                                      ///< os - output feature maps slice, i - input feature maps,
                                                      ///< yx - spatials, sv32 - 32 values of single slice.
        os_iyx_osv64,                                 ///< format used only for convolution weights:
                                                      ///< os - output feature maps slice, i - input feature maps,
                                                      ///< yx - spatials, sv64 - 64 values of single slice.
        image_2d_weights_c4_fyx_b,                    ///< image format for weights, width size is f*y*x/4
                                                      ///< (4-channels filled with fyx data), height is b
        image_2d_weights_c1_b_fyx,                    ///< image format for weights, width size is b,
                                                      ///< height is f*y*x, single channel
        winograd_2x3_s1_weights,                      ///< format used for weights for winograd non-fused
                                                      ///< convolution, F(2,3) -- filter 3x3 with stride 1
        winograd_2x3_s1_fused_weights,                ///< format used for weights for winograd fused
                                                      ///< convolution, F(2,3) -- filter 3x3 with stride 1
        winograd_6x3_s1_fused_weights,                ///< format used for weights for winograd fused
                                                      ///< convolution, F(6,3) -- filter 3x3 with stride 1
        image_2d_weights_winograd_6x3_s1_fbxyb,       ///< image format used for weights for winograd fused
                                                      ///< convolution, F(6,3) -- filter 3x3 with stride 1
        image_2d_weights_winograd_6x3_s1_xfbyb,       ///< image format used for weights for winograd fused
                                                      ///< convolution, F(6,3) -- filter 3x3 with stride 1
        os_is_yx_isa8_osv8_isv4,                      ///< format for weights for MMAD convolution
        os_is_zyx_isa8_osv8_isv4,                     ///< format for weights for MMAD convolution
        os_is_yx_isa8_osv16_isv4,                     ///< format for weights for fully connected MMAD
        os_is_zyx_isa8_osv16_isv4,                    ///< format for weights for fully connected MMAD
        os_is_yx_isa8_osv8_isv4_swizzled_by_4,        ///< format for weights for MMAD convolution
        os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4,   ///< format for weights for MMAD fsv32 convolution
        os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4,  ///< format for weights for MMAD fsv32 convolution
        is_o_yx_isv32,                                ///< format for weights for 1x1 MMAD convolutions
        is_o32_yx_isv32_swizzled_by_4,                ///< format for weights for 1x1 MMAD convolutions
        os_is_y_x8_osv8_isv4,                         ///< format for weights for 1x1 MMAD convolutions
        os_is_y_x8_osv8_isv4_swizzled_by_4,           ///< format for weights for 1x1 MMAD convolutions
        os_is_yx_osv16_isv4,                          ///< format for weights for IMAD convolutions
        os_is_zyx_osv16_isv16,                        ///< format for weights for IMAD convolutions
        os_is_yx_osv32_isv4_swizzled_by_2,            ///< format for weights for IMAD convolutions
        os_is_yx_osv32_isv4,                          ///< format for weights for IMAD convolutions
        os_is_zyx_osv32_isv4,                         ///< format for weights for IMAD convolutions
        os_is_yx_osv32_isv32p,                        ///< format for weights for binary convolutions
        lstm_weights_dio,                             ///< dynamic_lstm, direction,
                                                      ///< than IO (I - input size, O - 4 * hidden_size)
        os_is_osv32_isv32_swizzled_by_4,              ///< format for weights for 1x1 IMAD convolution
        os_iyx_osv32__ai32,
        iy_xs_os_xsv2_osv8__ao32,
        iy_xs_os_xsv2_osv16__ao32,
        i_yxs_os_yxsv2_osv16,
        os_i_yxs_osv4_yxsv4,

        goiyx,                                        ///< format used for weights for 2D convolution
        gioyx,                                        ///< format used for weights for 2D deconvolution
        yxiog,                                        ///< format used for weights for 2D convolution
        gyxio,                                        ///< format used for weights for 2D convolution
        goizyx,                                       ///< format used for weights for 3D convolution
        giozyx,                                       ///< format used for weights for 3D deconvolution
        g_os_iyx_osv16,                               ///< format used for weights for 2D convolution
        g_os_iyx_osv32,                               ///< format used for weights for 2D convolution
        gs_oiyx_gsv16,                                ///< format used for weights for 2D convolution
        gs_oizyx_gsv16,                               ///< format used for weights for 3D convolution
        gs_oiyx_gsv32,                                ///< format used for weights for 2D convolution
        g_is_os_zyx_isv16_osv16,                      ///< format used for grouped weights for blocked 3D deconvolution
        g_os_is_yx_osv16_isv4,
        g_os_is_zyx_osv16_isv16,
        g_is_os_yx_isv16_osv16,
        g_os_is_zyx_isv8_osv16_isv2,
        g_os_is_yx_isv8_osv16_isv2,
        g_os_is_zyx_isv16_osv16,
        g_os_zyx_is_osv16_isv4,                       ///< format for imad deconvolution
        g_os_zyx_is_osv16_isv16,                      ///< format for imad deconvolution
        g_os_zyx_is_osv16_isv32,                      ///< format for imad deconvolution
        g_os_zyx_is_osv32_isv4,                       ///< format for imad deconvolution
        g_os_zyx_is_osv32_isv16,                      ///< format for imad deconvolution
        g_os_zyx_is_osv32_isv32,                      ///< format for imad deconvolution
        g_os_is_yx_isv16_osv16,
        gs_oi_yxs_gsv4_yxsv4,
        gs_oi_yxs_gsv16_yxsv4,
        gs_oi_yxs_gsv32_yxsv4,
        gi_yxs_os_yxsv2_osv16,
        giy_xs_os_xsv2_osv8__ao32,
        giy_xs_os_xsv2_osv16__ao32,

        format_num,  ///< number of format types
        any        = -1
    };

    /// @brief Get format traits for particular @p format::type
    static const format_traits& traits(type fmt) {
        static const std::map<type, format_traits> traits {
                // B - number of Batch dimensions
                // F - number of Feature dimensions
                // S - number of Spatial dimensions
                // L - number of Local dimensions
                // G - number of Group dimensions
                // Order - dims changing order from rare to often
                // Inner order - dims order for internal storage in _sizes array
                // Block sizes - vector of pairs of dimension number (by inner order) and block size ordered from rare to often
                // Format                  B  F  S  L  G  Order  Inner order  Block sizes
                { yxfb,                  { 1, 1, 2, 0, 0, "yxfb",   "bfxy?",  {}}},
                { byxf,                  { 1, 1, 2, 0, 0, "byxf",   "bfxy?",  {}}},
                { bfyx,                  { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {}}},
                { fyxb,                  { 1, 1, 2, 0, 0, "fyxb",   "bfxy?",  {}}},
                { b_fs_yx_fsv16,         { 1, 1, 2, 0, 0, "bfyx",   "bfxy",   {{1, 16}}}},
                { b_fs_yx_fsv32,         { 1, 1, 2, 0, 0, "bfyx",   "bfxy",   {{1, 32}}}},
                { b_fs_zyx_fsv32,        { 1, 1, 3, 0, 0, "bfzyx",  "bfxyz",  {{1, 32}}}},
                { bs_xs_xsv8_bsv8,       { 1, 1, 1, 0, 0, "bx",     "b?x??",  {{2, 8}, {0, 8}}}},
                { bs_xs_xsv8_bsv16,      { 1, 1, 1, 0, 0, "bx",     "b?x??",  {{2, 8}, {0, 16}}}},
                { bs_x_bsv16,            { 1, 1, 1, 0, 0, "bx",     "b?x??",  {{0, 16}}}},
                { winograd_2x3_s1_data,  { 1, 1, 2, 0, 0, "bxyf",   "bfxy?",  {}}},
                { b_fs_yx_fsv4,          { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {{1, 4}}}},
                { bfzyx,                 { 1, 1, 3, 0, 0, "bfzyx",  "bfxyz",  {}}},
                { bfwzyx,                { 1, 1, 4, 0, 0, "bfwzyx", "bfxyzw", {}}},
                { fs_b_yx_fsv32,         { 1, 1, 2, 0, 0, "fbyx",   "bfxy?",  {{1, 32}}}},
                { b_fs_yx_32fp,          { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {}}},
                { b_fs_zyx_fsv16,        { 1, 1, 3, 0, 0, "bfzyx",  "bfxyz",  {{1, 16}}}},
                { bs_fs_zyx_bsv16_fsv16, { 1, 1, 3, 0, 0, "bfzyx",  "bfxyz",  {{0, 16 }, {1, 16}}}},
                { bs_fs_yx_bsv16_fsv16,  { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {{0, 16 }, {1, 16}}}},
                { nv12,                  { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {}}},
                { image_2d_rgba,         { 1, 1, 2, 0, 0, "bfyx",   "bfxy?",  {}}},

                { oiyx,                                        { 1, 1, 2, 0, 0, "oiyx",   "oixy",       {}}},
                { ioyx,                                        { 1, 1, 2, 0, 0, "ioyx",   "oixy",       {}}},
                { iyxo,                                        { 1, 1, 2, 0, 0, "iyxo",   "oixy",       {}}},
                { yxio,                                        { 1, 1, 2, 0, 0, "yxio",   "oixy?",      {}}},
                { oizyx,                                       { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {}}},
                { iozyx,                                       { 1, 1, 3, 0, 0, "iozyx",  "oixyz",      {}}},
                { os_is_yx_isv16_osv16,                        { 1, 1, 2, 0, 0, "oiyx",   "oixy",       {{1, 16}, {0, 16}}}},
                { os_iyx_osv16,                                { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {{0, 16}}}},
                { os_iyx_osv32,                                { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {{0, 32}}}},
                { os_iyx_osv64,                                { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {{0, 64}}}},
                { winograd_2x3_s1_weights,                     { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {}}},
                { winograd_2x3_s1_fused_weights,               { 1, 1, 2, 0, 0, "xyio",   "oixy?",      {}}},
                { winograd_6x3_s1_fused_weights,               { 1, 1, 2, 0, 0, "xyio",   "oixy?",      {}}},
                { image_2d_weights_winograd_6x3_s1_fbxyb,      { 1, 1, 2, 0, 0, "xyio",   "oixy?",      {}}},
                { image_2d_weights_winograd_6x3_s1_xfbyb,      { 1, 1, 2, 0, 0, "xyio",   "oixy?",      {}}},
                { image_2d_weights_c4_fyx_b,                   { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {}}},
                { image_2d_weights_c1_b_fyx,                   { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {}}},
                { lstm_weights_dio,                            { 1, 1, 2, 0, 0, "oixy",   "oixy?",      {}}},
                { os_is_yx_isa8_osv8_isv4,                     { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {}}},
                { os_is_yx_isa8_osv16_isv4,                    { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {}}},
                { os_is_yx_isa8_osv8_isv4_swizzled_by_4,       { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {}}},
                { os_is_zyx_isa8_osv8_isv4,                    { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{1, 8}, {0, 8}, {1, 4}}}},
                { os_is_zyx_isa8_osv16_isv4,                   { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{1, 8}, {0, 16}, {1, 4}}}},
                { os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4,  { 1, 1, 2, 0, 0, "oiyx",   "oixy?",      {{0, 32}, {1, 32}}}},
                { os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4, { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{0, 32}, {1, 32}}}},
                { is_o_yx_isv32,                               { 1, 1, 2, 0, 0, "oyxi",   "oixy?",      {{1, 32}}}},
                { is_o32_yx_isv32_swizzled_by_4,               { 1, 1, 2, 0, 0, "oyxi",   "oixy?",      {}}},
                { os_is_y_x8_osv8_isv4,                        { 1, 1, 2, 0, 0, "oyxi",   "oixy?",      {}}},
                { os_is_y_x8_osv8_isv4_swizzled_by_4,          { 1, 1, 2, 0, 0, "oyxi",   "oixy?",      {}}},
                { os_is_yx_osv16_isv4,                         { 1, 1, 2, 0, 0, "oixy",   "oixy?",      {{0, 16}, {1, 4}}}},
                { os_is_zyx_osv16_isv16,                       { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{0, 16}, {1, 16}}}},
                { os_is_yx_osv32_isv4_swizzled_by_2,           { 1, 1, 2, 0, 0, "oixy",   "oixy?",      {{0, 32}, {1, 4}}}},
                { os_is_yx_osv32_isv4,                         { 1, 1, 2, 0, 0, "oixy",   "oixy?",      {{0, 32}, {1, 4}}}},
                { os_is_zyx_osv32_isv4,                        { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{0, 32}, {1, 4}}}},
                { os_is_yx_osv32_isv32p,                       { 1, 1, 1, 0, 0, "oixy",   "oixy?",      {}}},
                { os_is_zyx_isv16_osv16,                       { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{0, 16}, {1, 16}}}},
                { is_os_zyx_isv16_osv16,                       { 1, 1, 3, 0, 0, "iozyx",  "oixyz",      {{1, 16}, {0, 16}}}},
                { is_os_yx_isv16_osv16,                        { 1, 1, 2, 0, 0, "ioyx",   "oixyz",      {{1, 16}, {0, 16}}}},
                { os_is_osv32_isv32_swizzled_by_4,             { 1, 1, 0, 0, 0, "oixy",   "oixy?",      {{0, 32}, {1, 32}}}},
                { os_is_zyx_isv8_osv16_isv2,                   { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{1, 8}, {0, 16}, {1, 2}}}},
                { os_zyxi_osv16,                               { 1, 1, 3, 0, 0, "ozyxi",  "oixyz",      {{0, 16}}}},
                { os_is_yx_isv8_osv16_isv2,                    { 1, 1, 2, 0, 0, "oizyx",  "oixyz",      {{1, 8}, {0, 16}, {1, 2}}}},
                { os_is_yx_osv16_isv16,                        { 1, 1, 2, 0, 0, "oiyx",   "oixy",       {{1, 16}, {0, 16}}}},
                { os_is_zyx_osv32_isv16,                       { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{0, 32}, {1, 16}}}},
                { os_is_zyx_osv64_isv16,                       { 1, 1, 3, 0, 0, "oizyx",  "oixyz",      {{0, 64}, {1, 16}}}},
                { os_iyx_osv32__ai32,                          { 1, 1, 2, 0, 0, "oiyx",   "oixy",       {{0, 32}}}},
                { i_yxs_os_yxsv2_osv16,                        { 1, 1, 2, 0, 0, "iyxo",   "oixy",       {{0, 16}}}},
                { iy_xs_os_xsv2_osv8__ao32,                    { 1, 1, 2, 0, 0, "iyxo",   "oixy",       {{2, 2}, {0, 8}}}},
                { iy_xs_os_xsv2_osv16__ao32,                   { 1, 1, 2, 0, 0, "iyxo",   "oixy",       {{2, 2}, {0, 16}}}},
                { os_i_yxs_osv4_yxsv4,                         { 1, 1, 2, 0, 0, "oiyx",   "oixy",       {{0, 4}}}},

                { goiyx,                                       { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {}}},
                { gioyx,                                       { 1, 1, 2, 0, 1, "gioyx",  "oixy????g",  {}}},
                { goizyx,                                      { 1, 1, 3, 0, 1, "goizyx", "oixyz???g",  {}}},
                { giozyx,                                      { 1, 1, 3, 0, 1, "giozyx", "oixyz???g",  {}}},
                { g_os_iyx_osv16,                              { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{0, 16}}}},
                { g_os_iyx_osv32,                              { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{0, 32}}}},
                { gs_oiyx_gsv16,                               { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{8, 16}}}},
                { gs_oizyx_gsv16,                              { 1, 1, 3, 0, 1, "goizyx", "oixyz???g",  {{8, 16}}}},
                { gs_oiyx_gsv32,                               { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{8, 32}}}},
                { gyxio,                                       { 1, 1, 2, 0, 1, "gyxio",  "oixy????g",  {}}},
                { g_is_os_zyx_isv16_osv16,                     { 1, 1, 3, 0, 1, "giozyx", "oixyz???g",  {{1, 16}, {0, 16}}}},
                { g_is_os_yx_isv16_osv16,                      { 1, 1, 2, 0, 1, "gioyx",  "oixy????g",  {{1, 16}, {0, 16}}}},
                { g_os_is_zyx_isv8_osv16_isv2,                 { 1, 1, 3, 0, 1, "goizyx", "oixyz???g",  {{1, 8}, {0, 16}, {1, 2}}}},
                { g_os_is_yx_isv8_osv16_isv2,                  { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{1, 8}, {0, 16}, {1, 2}}}},
                { g_os_is_zyx_isv16_osv16,                     { 1, 1, 3, 0, 1, "goizyx", "oixyz???g",  {{0, 16}, {1, 16}}}},
                { g_os_is_yx_osv16_isv4,                       { 1, 1, 2, 0, 1, "goixy",  "oixy????g",  {{0, 16}, {1, 4}}}},
                { g_os_is_zyx_osv16_isv16,                     { 1, 1, 3, 0, 1, "goizyx", "oixyz???g",  {{0, 16}, {1, 16}}}},
                { g_os_zyx_is_osv16_isv4,                      { 1, 1, 3, 0, 1, "gozyxi", "oixyz???g",  {{0, 16}, {1, 4}}}},
                { g_os_zyx_is_osv16_isv16,                     { 1, 1, 3, 0, 1, "gozyxi", "oixyz???g",  {{0, 16}, {1, 16}}}},
                { g_os_zyx_is_osv16_isv32,                     { 1, 1, 3, 0, 1, "gozyxi", "oixyz???g",  {{0, 16}, {1, 32}}}},
                { g_os_zyx_is_osv32_isv4,                      { 1, 1, 3, 0, 1, "gozyxi", "oixyz???g",  {{0, 32}, {1, 4}}}},
                { g_os_zyx_is_osv32_isv16,                     { 1, 1, 3, 0, 1, "gozyxi", "oixyz???g",  {{0, 32}, {1, 16}}}},
                { g_os_zyx_is_osv32_isv32,                     { 1, 1, 3, 0, 1, "gozyxi", "oixyz???g",  {{0, 32}, {1, 32}}}},
                { gs_oi_yxs_gsv4_yxsv4,                        { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{8, 4}}}},
                { gs_oi_yxs_gsv16_yxsv4,                       { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{8, 16}}}},
                { gs_oi_yxs_gsv32_yxsv4,                       { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{8, 32}}}},
                { g_os_is_yx_isv16_osv16,                      { 1, 1, 2, 0, 1, "goiyx",  "oixy????g",  {{1, 16}, {0, 16}}}},
                { gi_yxs_os_yxsv2_osv16,                       { 1, 1, 2, 0, 1, "giyxo",  "oixy????g",  {{0, 16}}}},
                { giy_xs_os_xsv2_osv8__ao32,                   { 1, 1, 2, 0, 1, "giyxo",  "oixy????g",  {{2, 2}, {0, 8}}}},
                { giy_xs_os_xsv2_osv16__ao32,                  { 1, 1, 2, 0, 1, "giyxo",  "oixy????g",  {{2, 2}, {0, 16}}}},
        };
        if (traits.find(fmt) == traits.end()) {
            throw std::runtime_error("[clDNN] Format description is missing in fmt traits");
        }
        return traits.at(fmt);
    }

    /// @brief Returns number of batch dimensions for a @p format.
    static size_t batch_num(type fmt) { return traits(fmt).batch_num; }
    /// @brief Returns number of feature dimensions for a @p format.
    static size_t feature_num(type fmt) { return traits(fmt).feature_num; }
    /// @brief Returns number of spatial dimensions for a @p format.
    static size_t spatial_num(type fmt) { return traits(fmt).spatial_num; }
    /// @brief Returns number of local dimensions for a @p format.
    static size_t local_num(type fmt) { return traits(fmt).local_num; }
    /// @brief Returns number of group dimensions for a @p format.
    static size_t group_num(type fmt) { return traits(fmt).group_num; }
    /// @brief Returns an order of dimensions for a @ format.
    static const std::string& order(type fmt) { return traits(fmt).order; }
    /// @brief Returns an internal orders of dimensions for a @p format.
    static const std::string& internal_order(type fmt) { return traits(fmt).internal_order; }
    /// @brief Returns block sizes for @p format.
    static const std::vector<std::pair<size_t, int>>& block_sizes(type fmt) { return traits(fmt).block_sizes; }
    /// @brief Returns number of dimensions contained within a @p format
    static size_t dimension(type fmt) { return order(fmt).size(); }
    /// @brief Checks if @p format is a winograd format
    static bool is_winograd(type fmt) {
        return (fmt == winograd_2x3_s1_data ||
                fmt == winograd_2x3_s1_weights ||
                fmt == winograd_2x3_s1_fused_weights ||
                fmt == winograd_6x3_s1_fused_weights ||
                fmt == image_2d_weights_winograd_6x3_s1_fbxyb ||
                fmt == image_2d_weights_winograd_6x3_s1_xfbyb); }
    /// @brief Checks if @p format is of image2d type
    static bool is_image_2d(type fmt) {
        return (fmt == image_2d_weights_c4_fyx_b ||
                fmt == image_2d_weights_c1_b_fyx ||
                fmt == image_2d_weights_winograd_6x3_s1_fbxyb ||
                fmt == image_2d_weights_winograd_6x3_s1_xfbyb ||
                fmt == nv12 ||
                fmt == image_2d_rgba);
    }
    /// @brief Checks if @p format is weights format
    static bool is_weights_format(type fmt) {
        const auto internal_order = traits(fmt).internal_order;
        const auto weights_chars = { "o", "i" };
        for (const auto& c : weights_chars) {
            if (internal_order.find_first_of(c) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
    /// @brief Checks if @p format is simple data format
    static bool is_simple_data_format(type fmt) {
        return (fmt == yxfb || fmt == byxf ||
                fmt == bfyx || fmt == fyxb ||
                fmt == bfzyx || fmt == bfwzyx);
    }
    /// @brief Checks if @p format is of grouped type
    static bool is_grouped(type fmt) { return group_num(fmt) != 0; }
    /// @brief Checks if @p format is of image type
    static bool is_image(type fmt) { return (is_image_2d(fmt)); }
    /// @brief Checks if @p format is blocked format
    static bool is_blocked(type fmt) { return !(block_sizes(fmt).empty()); }
    /// @brief Checks if @p format is nv12 format
    static bool is_nv12(type fmt) { return (fmt == nv12); }

    /// @brief Returns number of batch dimensions.
    size_t batch_num() const { return traits(value).batch_num; }
    /// @brief Returns number of feature dimensions.
    size_t feature_num() const { return traits(value).feature_num; }
    /// @brief Returns number of spatial dimensions.
    size_t spatial_num() const { return traits(value).spatial_num; }
    /// @brief Returns number of local dimensions.
    size_t local_num() const { return traits(value).local_num; }
    /// @brief Returns number of group dimensions.
    size_t group_num() const { return traits(value).group_num; }
    /// @brief Returns an order of dimensions in form of string.
    const std::string& order() const { return traits(value).order; }
    /// @brief Returns an internal orders of dimensions form of string.
    const std::string& internal_order() const { return traits(value).internal_order; }
    /// @brief Returns block sizes as vector of pairs of dimension and block size for that dimension.
    const std::vector<std::pair<size_t, int>>& block_sizes() const { return traits(value).block_sizes; }
    /// @brief Returns number of dimensions contained within this format
    size_t dimension() const { return order(value).size(); }
    /// @brief Checks if @p format is a winograd format
    bool is_winograd() const { return is_winograd(value); }
    /// @brief Checks if @p format is of image 2d type
    bool is_image_2d() const { return is_image_2d(value); }
    /// @brief Checks if @p format is of image type
    bool is_image() const { return is_image(value); }
    /// @brief Checks if @p format is blocked format
    bool is_blocked() { return is_blocked(value); }
    /// @brief Checks if @p format is a nv12 format
    bool is_nv12() const { return is_nv12(value); }

    /// @brief Transforms dimension from internal order to external order
    size_t internal_to_external(size_t idx) const {
        auto index = order().find_first_of(internal_order()[idx]);
        if (index == std::string::npos)
            throw std::invalid_argument("Internal dimension index does not map to external index.");
        return index;
    }

    type value;
    /// @brief Implicit conversion from format::type.
    constexpr format(type t) : value(t) {}
    /// @brief Implicit conversion to format::type.
    constexpr operator type() const { return value; }
};

// TODO: Estimate performance of ngraph structures and
// add corresponding cldnn impls for the classes below if needed

using dimension = ngraph::Dimension;
using rank = ngraph::Dimension;

class tensor : public ngraph::PartialShape {
public:
    using value_type = ngraph::Dimension::value_type;
    tensor(std::initializer_list<dimension> init) : ngraph::PartialShape(init) {}

    /// \brief Constructs a tensor with static rank from a vector of dimension.
    /// \param dimensions The dimension values for the constructed shape.
    tensor(std::vector<dimension> dimensions) : ngraph::PartialShape(dimensions) {}

    /// \brief Constructs a tensor with static rank from a vector of dimensions values.
    /// \param dimensions The dimension values for the constructed shape.
    tensor(const std::vector<dimension::value_type>& dimensions) : ngraph::PartialShape(dimensions) {}

    /// \brief Constructs a tensor with static rank from a vector of dimensions values.
    /// \param dimensions The dimension values for the constructed shape.
    // tensor(const std::vector<size_t>& dimensions) : ngraph::PartialShape(std::vector<dimension::value_type>(dimensions.begin(), dimensions.end())) {}

    tensor(ngraph::Rank r, value_type v) : ngraph::PartialShape(std::vector<dimension::value_type>(r.get_length(), v)) {}

    /// \brief Constructs a static tensor with zero rank (the shape of a scalar).
    tensor() : ngraph::PartialShape() {}

    explicit tensor(ngraph::PartialShape other) : ngraph::PartialShape(other) {}

    static tensor dynamic(ngraph::Rank r = ngraph::Rank::dynamic()) {
        return tensor(ngraph::PartialShape::dynamic(r));
    }

    // For backward compatibility purposes. Should be removed as soon as code usages are fixed
    explicit tensor(value_type v) : ngraph::PartialShape({0, 0, 0, 0}) {}
    tensor(format fmt, std::vector<dimension> dimensions, value_type init_val) : ngraph::PartialShape(dimensions) {}

    static tensor max(tensor a, tensor b) {
        ngraph::Rank dst_rank;
        if (!ngraph::Rank::merge(dst_rank, a.rank(), b.rank())) {
            throw std::runtime_error("[GPU] tensor::max error: ranks are not compatible");
        }
        if (a.is_dynamic() || b.is_dynamic()) {
            throw std::runtime_error("[GPU] tensor::max error: dynamic shapes are not supported");
        }

        std::vector<dimension::value_type> max_shape(dst_rank.get_length());
        for (auto i = 0; i < dst_rank.get_length(); i++) {
            max_shape[i] = std::max(a[i].get_length(), b[i].get_length());
        }
        return tensor(max_shape);
    }

    static tensor min(tensor a, tensor b) {
        ngraph::Rank dst_rank;
        if (!ngraph::Rank::merge(dst_rank, a.rank(), b.rank())) {
            throw std::runtime_error("[GPU] tensor::max error: ranks are not compatible");
        }
        if (a.is_dynamic() || b.is_dynamic()) {
            throw std::runtime_error("[GPU] tensor::max error: dynamic shapes are not supported");
        }

        std::vector<dimension::value_type> min_shape(dst_rank.get_length());
        for (auto i = 0; i < dst_rank.get_length(); i++) {
            min_shape[i] = std::min(a[i].get_length(), b[i].get_length());
        }
        return tensor(min_shape);
    }

    tensor negate() const {
        tensor result = *this;
        for (auto i = 0; i < result.rank().get_length(); i++) {
            result[i] = -result[i].get_length();
        }

        return result;
    }

    tensor add(const tensor& other) const {
        tensor result = *this;
        CLDNN_CHECK(other.rank() == this->rank(), "ranks must be equal");
        for (auto i = 0; i < result.rank().get_length(); i++) {
            result[i] = result[i] + other[i];
        }

        return result;
    }

    tensor sub(const tensor& other) const {
        tensor result = *this;
        CLDNN_CHECK(other.rank() == this->rank(), "ranks must be equal");
        for (auto i = 0; i < result.rank().get_length(); i++) {
            result[i] = result[i] - other[i];
        }

        return result;
    }

    std::vector<value_type> sizes(cldnn::format /*fmt*/) const {
        // FIXME[GPU_DYN]: Do we need to respect fmt here?
        return sizes();
    }

    /// @brief Assign and add
    tensor& operator+=(const tensor& rhs) {
        for (auto i = 0; i < this->rank().get_length(); i++) {
            (*this)[i] = (*this)[i] + rhs[i];
        }
        return *this;
    }

    /// @brief Assign and add
    tensor& operator-=(const tensor& rhs) {
        for (auto i = 0; i < this->rank().get_length(); i++) {
            (*this)[i] = (*this)[i] - rhs[i];
        }
        return *this;
    }

    std::vector<value_type> sizes() const {
        auto r = this->rank().get_length();
        std::vector<value_type> res(r);
        for (auto i = 0; i < r; i++) {
            res[i] = (*this)[i].get_length();
        }
        return res;
    }

    /// Number of elements to be stored in this memory layout
    size_t count() const {
        CLDNN_CHECK(is_static(), "count() can be called for static shape only");
        size_t size = 1;
        for (auto d : *this) {
            size *= d.get_length();
        }
        return size;
    }

    value_type get_dim(size_t idx) const {
        if (idx >= get_shape().size())
            throw std::runtime_error("[GPU] Invalid index in get_dim method");

        return this->get_shape()[idx];
    }

    // TODO(GPU_DYN) the methods below must be moved to layout
    value_type spatial(size_t i) const {
        return get_dim(2 + i);
    }

    void set_spatial(size_t i, value_type v) {
        (*this)[2 + i] = v;
    }

    value_type feature(size_t /* i */) const {
        return get_dim(1);
    }

    void set_feature(size_t /* i */, value_type v) {
        (*this)[1] = v;
    }

    value_type batch(size_t /* i */) const {
        return get_dim(0);
    }

    void set_batch(size_t /* i */, value_type v) {
        (*this)[0] = v;
    }

    value_type group(size_t /* i */) const {
        return get_dim(0);
    }

    void set_group(size_t /* i */, value_type v) {
        (*this)[0] = v;
    }

    std::string to_string() const {
        std::stringstream s;
        s << (*this);

        return s.str();
    }
};

#define TensorValue(x) static_cast<cldnn::tensor::value_type>(x)

/// @brief Adds two @p tensors
inline tensor operator+(const tensor& lhs, const tensor& rhs) { return lhs.add(rhs); }
/// @brief Subtracts two @p tensors
inline tensor operator-(const tensor& lhs, const tensor& rhs) { return lhs.sub(rhs); }

/// @}
/// @}
}  // namespace cldnn
