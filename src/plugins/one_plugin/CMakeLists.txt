# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "openvino_one_plugin")

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wno-strict-aliasing)
endif()

if(OV_COMPILER_IS_CLANG)
    ov_add_compiler_flags(-Wno-delete-non-abstract-non-virtual-dtor)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # 4267 4244 conversion from 'XXX' to 'YYY', possible loss of data
    ov_add_compiler_flags(/wd4244)
    # '<': signed/unsigned mismatch
    ov_add_compiler_flags(/wd4018)
endif()

# if(ENABLE_GPU_DEBUG_CAPS)
#     add_definitions(-DGPU_DEBUG_CONFIG=1)
# endif()

set(INTEL_GPU_TARGET_OCL_VERSION "200" CACHE STRING "Target version of OpenCL which should be used by GPU plugin")

add_definitions(-DCL_TARGET_OPENCL_VERSION=${INTEL_GPU_TARGET_OCL_VERSION})

set(MAIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# add_subdirectory(thirdparty)
# include(thirdparty/cmake/rapidjson.cmake)

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Werror)
endif()

# add_subdirectory(src/runtime)
# add_subdirectory(src/kernel_selector)
# add_subdirectory(src/graph)

file(GLOB PLUGIN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/opset/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/impls/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/extension/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/*.hpp
        )


set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/")
add_subdirectory(src/backend/cpu)
add_subdirectory(src/backend/ocl)

list(APPEND OBJ_LIBS cpu_impls)
list(APPEND OBJ_LIBS ocl_impls)
ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "ONE"
              SOURCES ${PLUGIN_SOURCES}
              DEFAULT_CONFIG ${PLUGIN_DEFAULT_CONFIG}
              OBJECT_LIBRARIES ${OBJ_LIBS}
              VERSION_DEFINES_FOR src/plugin/plugin.cpp)

target_compile_options(${TARGET_NAME} PRIVATE
    $<$<CONFIG:Release>:$<IF:$<CXX_COMPILER_ID:MSVC>,/Os,-Os>>)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::pugixml openvino::shape_inference openvino::runtime openvino::runtime::dev)

target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        ${CMAKE_CURRENT_SOURCE_DIR}/src/)

ov_set_threading_interface_for(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# Workaround to avoid warnings during LTO build
if(CMAKE_COMPILER_IS_GNUCXX)
  set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS_RELEASE "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized"
                                                  LINK_FLAGS_RELWITHDEBINFO "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized")
endif()

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()

# Failed because of OpenCL
# must be called after all target_link_libraries
# ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
