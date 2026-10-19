# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.21.0)
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(mps4_timer_tests LANGUAGES C)
    include(CTest)
endif()

get_filename_component(MLEK_TIMER_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

foreach(TIMER_PLATFORM IN ITEMS fvp fpga)
    set(TIMER_TEST_TARGET mps4_timer_${TIMER_PLATFORM}_tests)
    add_executable(${TIMER_TEST_TARGET} Mps4TimerTests.c)
    target_compile_features(${TIMER_TEST_TARGET} PRIVATE c_std_11)
    target_compile_definitions(${TIMER_TEST_TARGET} PRIVATE CPU_PROFILE_ENABLED=1)
    if (TIMER_PLATFORM STREQUAL "fpga")
        target_compile_definitions(${TIMER_TEST_TARGET} PRIVATE FPGA=1)
    endif()
    target_include_directories(${TIMER_TEST_TARGET} PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${MLEK_TIMER_REPO_ROOT}/source/hal/source/platform/mps4/include
        ${MLEK_TIMER_REPO_ROOT}/source/hal/source/components/platform_pmu/include
        ${MLEK_TIMER_REPO_ROOT}/source/hal/source/components/log)
    add_test(NAME mps4-timer-${TIMER_PLATFORM} COMMAND ${TIMER_TEST_TARGET})
endforeach()
