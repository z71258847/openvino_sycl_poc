# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ocl_impls")

find_host_package(Python3 REQUIRED COMPONENTS Interpreter)

file(GLOB_RECURSE LIBRARY_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )

file(GLOB_RECURSE KERNELS
    "${CMAKE_CURRENT_SOURCE_DIR}/cl_kernels/*.cl"
  )

# Path which points to root directory where code generated elements are created
# (specific to build configuration).
set(CODEGEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/codegen")
set(CODEGEN_CACHE_DIR  "${CODEGEN_DIR}/cache")

# Path which points to automatically included directory with code generated elements
# (to support "copy-if-different" optimization).
set(CODEGEN_INCDIR  "${CODEGEN_DIR}/include")


set(PRIM_DB "kernels_db.inc")
set(PRIM_DB_BATCH_HEADERS "kernels_db_batch_headers.inc")
set(CODEGEN_CACHE_SOURCES "${CODEGEN_INCDIR}/${PRIM_DB}"
                          "${CODEGEN_INCDIR}/${PRIM_DB_BATCH_HEADERS}")

set(CODEGEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/common/kernels_db_gen.py")
# Helping with some generators.
set_property(SOURCE ${CODEGEN_CACHE_SOURCES} PROPERTY GENERATED TRUE)

add_custom_command(OUTPUT "${CODEGEN_CACHE_DIR}/${PRIM_DB}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CODEGEN_CACHE_DIR}"
  COMMAND "${Python3_EXECUTABLE}" "${CODEGEN_SCRIPT}" -out_path "${CODEGEN_CACHE_DIR}"
                                                      -out_file_name_prim_db "${PRIM_DB}"
                                                      -out_file_name_batch_headers "${PRIM_DB_BATCH_HEADERS}"
                                                      -kernels "${CMAKE_CURRENT_SOURCE_DIR}/cl_kernels/"
  DEPENDS ${KERNELS} "${CODEGEN_SCRIPT}"
  COMMENT "Generating ${CODEGEN_CACHE_DIR}/${PRIM_DB} ..."
)

add_custom_command(OUTPUT "${CODEGEN_INCDIR}/${PRIM_DB}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CODEGEN_CACHE_DIR}/${PRIM_DB}" "${CODEGEN_INCDIR}/${PRIM_DB}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CODEGEN_CACHE_DIR}/${PRIM_DB_BATCH_HEADERS}" "${CODEGEN_INCDIR}/${PRIM_DB_BATCH_HEADERS}"
  DEPENDS "${CODEGEN_CACHE_DIR}/${PRIM_DB}" "${KERNELS}" "${CODEGEN_SCRIPT}"
  COMMENT "Updating file if the file changed (${PRIM_DB}) ..."
)

add_library(${TARGET_NAME} OBJECT ${LIBRARY_SRC} ${CODEGEN_CACHE_SOURCES})

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${INCLUDE_DIR}>
                                                 $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                 $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kernels/>
                                                 $<BUILD_INTERFACE:${CODEGEN_INCDIR}>
                                                 ${ROOT_DIR}
                                          )


target_link_libraries(${TARGET_NAME} PUBLIC OpenCL::OpenCL openvino::runtime openvino::runtime::dev)
