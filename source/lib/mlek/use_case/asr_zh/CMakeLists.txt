# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.21)
add_library(asr_zh_api STATIC KaldiFbank.cc CtcDecoder.cc StreamingState.cc)
target_link_libraries(asr_zh_api PUBLIC common_api)
add_library(mlek::asr_zh_api ALIAS asr_zh_api)

