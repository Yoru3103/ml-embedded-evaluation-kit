# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.21)
project(asr_vsi_tests LANGUAGES CXX)

include(CTest)
option(ASR_VSI_TEST_SANITIZERS "Enable address and undefined-behavior sanitizers" ON)
get_filename_component(MLEK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(asr_vsi_vad_tests VadTests.cc)
add_executable(asr_vsi_custom_vad_tests CustomVadTests.cc)
target_compile_definitions(asr_vsi_custom_vad_tests PRIVATE
    ASR_VSI_VAD_RMS=400 ASR_VSI_SILENCE_MS=1000)

# Compile the production handler against native transport/model/display mocks.
# No HAL, ML framework, core-platform, or firmware libraries are linked.
add_executable(asr_vsi_handler_tests HandlerTests.cc
    ${MLEK_ROOT}/source/app/use_case/asr/src/ConformerVsiHandler.cc)
target_include_directories(asr_vsi_handler_tests BEFORE PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/mocks
    ${MLEK_ROOT}/source/app/audio_demo/include)
target_compile_definitions(asr_vsi_handler_tests PRIVATE ASR_INPUT_VSI)

foreach(target asr_vsi_vad_tests asr_vsi_custom_vad_tests asr_vsi_handler_tests)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_include_directories(${target} PRIVATE
        ${MLEK_ROOT}/source/app/use_case/asr/include)
    # Assertions execute test actions, so retain them in Release builds too.
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /UNDEBUG)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -UNDEBUG)
        if(ASR_VSI_TEST_SANITIZERS)
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined
                -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
endforeach()

add_test(NAME asr-vsi-vad COMMAND asr_vsi_vad_tests)
add_test(NAME asr-vsi-custom-vad COMMAND asr_vsi_custom_vad_tests)
add_test(NAME asr-vsi-handler COMMAND asr_vsi_handler_tests)
