// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <vector>
#include <array>
#include <algorithm>
#include <type_traits>
#include <string>
#include <utility>


namespace cldnn {

#define CLDNN_CHECK(condition, add_msg) \
    cldnn_check(__FILE__, __LINE__, #condition, (condition), add_msg)

void cldnn_check(const std::string& file,
                   int line,
                   std::string condition_id,
                   bool condition,
                   const std::string& additional_message = "");

}  // namespace cldnn
