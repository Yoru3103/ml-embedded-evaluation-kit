# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
# Run: cmake -DPYTHON=<venv-python> -P tests/windows/source_generator_smoke.cmake
if(NOT DEFINED PYTHON)
    message(FATAL_ERROR "Pass -DPYTHON=<host Python executable>")
endif()
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
# Exercise path-list preservation, including a Windows semicolon.
if(CMAKE_HOST_WIN32)
    set(ENV{PYTHONPATH} "C:/nonexistent first;C:/nonexistent second")
else()
    set(ENV{PYTHONPATH} "/nonexistent first:/nonexistent second")
endif()
include("${REPO_ROOT}/scripts/cmake/source_gen_utils.cmake")
set(OUTPUT_DIR "${REPO_ROOT}/build/windows-generator-smoke/with spaces")
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
generate_labels_code(
    INPUT "${REPO_ROOT}/resources/img_class/labels/labels_mobilenet_v2_1.0_224.txt"
    DESTINATION_SRC "${OUTPUT_DIR}"
    DESTINATION_HDR "${OUTPUT_DIR}"
    OUTPUT_FILENAME SmokeLabels)
if(NOT EXISTS "${OUTPUT_DIR}/SmokeLabels.cc" OR
        NOT EXISTS "${OUTPUT_DIR}/SmokeLabels.hpp")
    message(FATAL_ERROR "Expected generated labels were not produced")
endif()
message(STATUS "Source generator smoke test passed")
