# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_CXX_COMPILER /home/vparamuz/intel/oneapi/compiler/2022.0.2/linux/bin-llvm/clang++)
set(CMAKE_C_COMPILER /home/vparamuz/intel/oneapi/compiler/2022.0.2/linux/bin-llvm/clang)
set(CMAKE_CXX_FALGS "" CACHE STRING "" FORCE)
set(CMAKE_C_FALGS "" CACHE STRING "" FORCE)

set (CMAKE_CXX_STANDARD 17)

ie_add_sample(NAME classification_sample_async
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/classification_sample_async.h"
              DEPENDENCIES format_reader ie_samples_utils)

target_compile_options(classification_sample_async PUBLIC -fsycl)
target_link_options(classification_sample_async PUBLIC -fsycl)
target_link_libraries(classification_sample_async PUBLIC OpenCL)
