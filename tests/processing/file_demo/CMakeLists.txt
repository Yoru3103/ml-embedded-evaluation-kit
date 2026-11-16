# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.21)
project(file_demo_tests LANGUAGES CXX)
enable_testing()
add_executable(file_input_tests FileInputTests.cc)
target_compile_features(file_input_tests PRIVATE cxx_std_17)
target_include_directories(file_input_tests PRIVATE mocks
    ../../../source/app/file_demo/include)
if (MSVC)
    target_compile_options(file_input_tests PRIVATE /W4 /UNDEBUG)
else()
    target_compile_options(file_input_tests PRIVATE -Wall -Wextra -Werror -UNDEBUG)
endif()
add_test(NAME uart-file-input COMMAND file_input_tests)
