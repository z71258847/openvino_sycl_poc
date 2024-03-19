// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
class TRANSFORMATIONS_API DeReshapeConcat;
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Transformation uses symbol / label information to optimize out Reshape operations surrounding special cases of
 * Concat. It checks that surrounding Reshapes are not touching axis dimension of concat and out reshape produces same shape sheme as it
 * was originally before input reshapes.
 */
class ov::pass::DeReshapeConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DeReshapeConcat", "0");
    DeReshapeConcat();
};
