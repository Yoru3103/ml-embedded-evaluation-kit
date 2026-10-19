# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.21.0)
project(profiler_counter_tests LANGUAGES CXX)
include(CTest)
get_filename_component(MLEK_PROFILER_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_executable(profiler_counter_tests ProfilerCounterTests.cc
    ${MLEK_PROFILER_ROOT}/source/app/profiler/Profiler.cc)
target_compile_features(profiler_counter_tests PRIVATE cxx_std_17)
target_include_directories(profiler_counter_tests PRIVATE
    include
    ${MLEK_PROFILER_ROOT}/source/app/profiler/include
    ${MLEK_PROFILER_ROOT}/source/hal/include
    ${MLEK_PROFILER_ROOT}/source/hal/source/components/platform_pmu/include
    ${MLEK_PROFILER_ROOT}/source/lib)
add_test(NAME profiler-counters COMMAND profiler_counter_tests)
