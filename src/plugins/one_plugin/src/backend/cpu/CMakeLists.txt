# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "cpu_impls")

file(GLOB_RECURSE LIBRARY_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )

add_library(${TARGET_NAME} OBJECT ${LIBRARY_SRC})

target_include_directories(${TARGET_NAME} PUBLIC ${ROOT_DIR})
target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime openvino::runtime::dev openvino::reference)
