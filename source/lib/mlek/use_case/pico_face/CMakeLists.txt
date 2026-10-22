# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.21.0)
add_library(pico_face_api STATIC PicoFaceSampler.cc PicoFaceProcessing.cc PicoFaceSerial.cc)
target_link_libraries(pico_face_api PUBLIC common_api)
add_library(mlek::pico_face_api ALIAS pico_face_api)
