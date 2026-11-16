# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.21)
project(fvp_audio_tests LANGUAGES CXX)
include(CTest)
add_executable(fvp_audio_tests AudioWindowTests.cc)
target_include_directories(fvp_audio_tests PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../source/app/audio_demo/include)
target_compile_features(fvp_audio_tests PRIVATE cxx_std_17)
if (MSVC)
    target_compile_options(fvp_audio_tests PRIVATE /UNDEBUG)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(fvp_audio_tests PRIVATE -Wall -Wextra -Werror -UNDEBUG
        -fsanitize=address,undefined)
    target_link_options(fvp_audio_tests PRIVATE -fsanitize=address,undefined)
endif()
add_test(NAME fvp-audio-windows COMMAND fvp_audio_tests)
