# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.21)
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(kws_zh_processing_tests LANGUAGES CXX)
    include(CTest)
endif()
get_filename_component(MLEK_TEST_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_executable(kws_zh_processing_tests
    KwsZhProcessingTests.cc
    ${MLEK_TEST_ROOT}/source/lib/mlek/use_case/kws_zh/KaldiFbank.cc
    ${MLEK_TEST_ROOT}/source/lib/mlek/use_case/kws_zh/WakeWordDecoder.cc
    ${MLEK_TEST_ROOT}/source/lib/mlek/math/PlatformMath.cc)
target_include_directories(kws_zh_processing_tests PRIVATE ${MLEK_TEST_ROOT}/source/lib)
target_compile_features(kws_zh_processing_tests PRIVATE cxx_std_17)
target_compile_definitions(kws_zh_processing_tests PRIVATE MLEK_LOG_ENABLE)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT CMAKE_CROSSCOMPILING)
    target_compile_options(kws_zh_processing_tests PRIVATE -fsanitize=address,undefined)
    target_link_options(kws_zh_processing_tests PRIVATE -fsanitize=address,undefined)
endif()
add_test(NAME kws-zh-processing COMMAND kws_zh_processing_tests)
