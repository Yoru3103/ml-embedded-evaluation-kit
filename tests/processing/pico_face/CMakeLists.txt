# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.21.0)
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(pico_face_processing_tests LANGUAGES CXX)
    include(CTest)
endif()
get_filename_component(MLEK_TEST_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_executable(pico_face_processing_tests PicoFaceTests.cc
    ${MLEK_TEST_ROOT}/source/lib/mlek/use_case/pico_face/PicoFaceSampler.cc
    ${MLEK_TEST_ROOT}/source/lib/mlek/use_case/pico_face/PicoFaceProcessing.cc)
target_include_directories(pico_face_processing_tests PRIVATE ${MLEK_TEST_ROOT}/source/lib)
target_compile_features(pico_face_processing_tests PRIVATE cxx_std_17)
add_test(NAME pico-face-processing COMMAND pico_face_processing_tests)

add_executable(pico_face_serial_fixture PicoFaceSerialFixture.cc
    ${MLEK_TEST_ROOT}/source/lib/mlek/use_case/pico_face/PicoFaceSerial.cc
    ${MLEK_TEST_ROOT}/source/lib/mlek/use_case/pico_face/PicoFaceProcessing.cc)
target_include_directories(pico_face_serial_fixture PRIVATE ${MLEK_TEST_ROOT}/source/lib)
target_compile_features(pico_face_serial_fixture PRIVATE cxx_std_17)
find_package(Python3 COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
    add_test(NAME pico-face-serial COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${MLEK_TEST_ROOT}/scripts/py"
        "PICO_FACE_FRAME_FIXTURE=$<TARGET_FILE:pico_face_serial_fixture>"
        ${Python3_EXECUTABLE} -m unittest discover
        -s ${MLEK_TEST_ROOT}/tests/py -p test_pico_face_serial.py)
endif()
