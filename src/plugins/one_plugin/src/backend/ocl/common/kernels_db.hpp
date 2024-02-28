// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <cctype>
#include <string>

/// \brief Class providing interface to retrieve a list of primitive implementations per primitive id
///
namespace ov {
namespace ocl {

using Code = std::string;
using KernelName = std::string;

struct KernelsDataBase {
    KernelsDataBase();

    std::vector<Code> get(const KernelName& id) const;
    std::vector<Code> get_batch_header_str() const { return std::move(batch_header_str); }

private:
    struct case_insensitive_compare {
        bool operator()(const KernelName& lhs, const KernelName& rhs) const {
            return std::lexicographical_compare(lhs.begin(),
                                                lhs.end(),
                                                rhs.begin(),
                                                rhs.end(),
                                                [](const char& a, const char& b) { return tolower(a) < tolower(b); });
        }
    };
    std::multimap<KernelName, Code, case_insensitive_compare> primitives;
    std::vector<Code> batch_header_str;
};

}  // namespace ocl
}  // namespace ov
