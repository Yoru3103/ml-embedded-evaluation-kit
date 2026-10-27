#----------------------------------------------------------------------------
#  SPDX-FileCopyrightText: Copyright 2021-2026 Arm Limited and/or its
#  affiliates <open-source-office@arm.com>
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.21.0)

# Build in release mode by default
if (NOT CMAKE_BUILD_TYPE STREQUAL Debug)
    set(CMAKE_BUILD_TYPE Release CACHE INTERNAL "")
endif()

message(STATUS "Build type is set to ${CMAKE_BUILD_TYPE}")

# Set language standards.
set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 17)

# Make the standard a requirement => prevent fallback to previous
# supported standard
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# We want to pass standard C/C++ flags, without gnu extensions
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Initialise global settings
include("${CMAKE_CURRENT_SOURCE_DIR}/MlekModule.cmake")

project(arm_ml_embedded_evaluation_kit
        VERSION         26.09.0
        DESCRIPTION     "ARM ML Embedded Evaluation Kit"
        LANGUAGES       C CXX ASM)

enforce_compiler_version()
setup_source_generator()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(ASR_STAGE_PROFILING "Enable optional wav2letter processing-stage timing" OFF)
option(ASR_SIMPLE_STAGE_PROFILING "Enable wav2letter six-stage timing" OFF)
option(ASR_ET_STAGE_PROFILING "Enable Conformer ExecuTorch six-stage timing" OFF)
if (ASR_STAGE_PROFILING AND ASR_SIMPLE_STAGE_PROFILING)
    message(FATAL_ERROR "Select either simple or detailed wav2letter timing, not both")
endif()
if (ASR_SIMPLE_STAGE_PROFILING)
    add_compile_definitions(ASR_SIMPLE_STAGE_PROFILING)
endif()
if (ASR_STAGE_PROFILING)
    add_compile_definitions(ASR_STAGE_PROFILING)
endif()

# Add hal target
add_subdirectory(source/hal ${CMAKE_BINARY_DIR}/mlek-hal)

if (ASR_ET_STAGE_PROFILING AND NOT CPU_PROFILE_ENABLED)
    message(FATAL_ERROR "ASR_ET_STAGE_PROFILING requires CPU_PROFILE_ENABLED=ON")
endif()

if ((ASR_STAGE_PROFILING OR ASR_SIMPLE_STAGE_PROFILING) AND NOT CPU_PROFILE_ENABLED)
    message(FATAL_ERROR "ASR stage profiling requires CPU_PROFILE_ENABLED=ON")
endif()

# Add platform agnostic lib
add_subdirectory(source/lib ${CMAKE_BINARY_DIR}/mlek-lib)

# Add app (and potentially tests)
add_subdirectory(source/app ${CMAKE_BINARY_DIR}/mlek-app)

if (TARGET_PLATFORM STREQUAL "native")
    add_subdirectory(tests/processing/asr ${CMAKE_BINARY_DIR}/asr-processing-tests)
    add_subdirectory(tests/hal/timer ${CMAKE_BINARY_DIR}/mps4-timer-tests)
    add_subdirectory(tests/hal/display ${CMAKE_BINARY_DIR}/mps4-display-tests)
    add_subdirectory(tests/processing/yolov8_detection ${CMAKE_BINARY_DIR}/yolov8-processing-tests)
    add_subdirectory(tests/processing/pico_face ${CMAKE_BINARY_DIR}/pico-face-processing-tests)
endif()

print_useroptions()
