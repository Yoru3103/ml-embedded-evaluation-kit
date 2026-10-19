# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.21.0)
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(yolov8_processing_tests LANGUAGES CXX)
    include(CTest)
endif()
get_filename_component(MLEK_TEST_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_executable(yolov8_processing_tests
    YoloV8ProcessingTests.cc
    ${MLEK_TEST_ROOT}/source/lib/mlek/use_case/yolov8_detection/YoloV8PreProcessing.cc
    ${MLEK_TEST_ROOT}/source/lib/mlek/use_case/yolov8_detection/YoloV8PostProcessing.cc)
target_include_directories(yolov8_processing_tests PRIVATE
    ${MLEK_TEST_ROOT}/source/lib
    ${MLEK_TEST_ROOT}/source/app/use_case/qrcode/include)
target_compile_features(yolov8_processing_tests PRIVATE cxx_std_17)
add_test(NAME yolov8-processing COMMAND yolov8_processing_tests)
