# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.21)
project(qrcode_demo_tests LANGUAGES CXX)
include(CTest)
add_executable(qrcode_request_tests QrCodeImageRequestTests.cc)
target_compile_features(qrcode_request_tests PRIVATE cxx_std_17)
target_include_directories(qrcode_request_tests PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../source/app/use_case/qrcode/include)
if(MSVC)
    target_compile_options(qrcode_request_tests PRIVATE /W4 /UNDEBUG)
else()
    target_compile_options(qrcode_request_tests PRIVATE -Wall -Wextra -Werror -UNDEBUG)
endif()
add_test(NAME qrcode-image-request COMMAND qrcode_request_tests)
