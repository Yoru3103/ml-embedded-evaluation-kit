# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.21.0)

# These framebuffer tests can also run without model or framework dependencies.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(mps4_display_tests LANGUAGES C CXX)
    include(CTest)
endif()

get_filename_component(MLEK_DISPLAY_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

foreach(PIXEL_BYTES IN ITEMS 3 4)
    set(DISPLAY_TEST_TARGET mps4_display_rgb${PIXEL_BYTES}_tests)
    add_executable(${DISPLAY_TEST_TARGET}
        Mps4DisplayTests.c
        DetectionDrawingTests.cc
        ${MLEK_DISPLAY_REPO_ROOT}/source/app/use_case/yolov8_detection/src/DetectionDrawing.cc)
    target_compile_features(${DISPLAY_TEST_TARGET} PRIVATE c_std_11 cxx_std_17)
    target_include_directories(${DISPLAY_TEST_TARGET} PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${MLEK_DISPLAY_REPO_ROOT}/source/lib
        ${MLEK_DISPLAY_REPO_ROOT}/source/app/use_case/yolov8_detection/include
        ${MLEK_DISPLAY_REPO_ROOT}/source/hal/source/components/display/include
        ${MLEK_DISPLAY_REPO_ROOT}/source/hal/source/components/log)
    # The real MPS4 HAL writes to a guarded array instead of physical MCU memory.
    target_compile_definitions(${DISPLAY_TEST_TARGET} PRIVATE
        __arm__=1
        __ARM_ARCH_PROFILE=77
        HDLCD_RES_WIDTH=8
        HDLCD_RES_HEIGHT=6
        HDLCD_BYTES_PER_PIXEL=${PIXEL_BYTES})
    add_test(NAME mps4-display-rgb${PIXEL_BYTES} COMMAND ${DISPLAY_TEST_TARGET})
endforeach()
