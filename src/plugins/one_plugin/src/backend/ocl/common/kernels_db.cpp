// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels_db.hpp"
#include <assert.h>
#include <algorithm>
#include <vector>
#include <utility>
#include <stdexcept>

#ifndef NDEBUG
#include <fstream>
#include <iostream>
#endif

namespace ov {
namespace ocl {

KernelsDataBase::KernelsDataBase()
    : primitives({
#include "kernels_db.inc"
      }),
      batch_header_str({
#include "kernels_db_batch_headers.inc"
      }) {
}

std::vector<Code> KernelsDataBase::get(const KernelName& id) const {
#ifndef NDEBUG
    {
        std::ifstream kernel_file{id + ".cl", std::ios::in | std::ios::binary};
        if (kernel_file.is_open()) {
            code ret;
            auto beg = kernel_file.tellg();
            kernel_file.seekg(0, std::ios::end);
            auto end = kernel_file.tellg();
            kernel_file.seekg(0, std::ios::beg);

            ret.resize((size_t)(end - beg));
            kernel_file.read(&ret[0], (size_t)(end - beg));

            return {std::move(ret)};
        }
    }
#endif
    try {
        const auto codes = primitives.equal_range(id);
        std::vector<Code> temp;
        std::for_each(codes.first, codes.second, [&](const std::pair<const std::string, std::string>& c) {
            temp.push_back(c.second);
        });

        if (temp.size() != 1) {
            throw std::runtime_error("cannot find the kernel " + id + " in primitive database.");
        }

        return temp;
    } catch (...) {
        throw std::runtime_error("cannot find the kernel " + id + " in primitive database.");
    }
}

}  // namespace ocl
}  // namespace ov
