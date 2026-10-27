# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.21)
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(conformer_processing_tests LANGUAGES CXX)
    include(CTest)
endif()
get_filename_component(MLEK_TEST_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_executable(conformer_processing_tests
    ConformerProcessingTests.cc
    ${MLEK_TEST_ROOT}/source/lib/mlek/use_case/asr/ConformerProcessing.cc
    ${MLEK_TEST_ROOT}/source/lib/mlek/use_case/asr/ConformerMelSpectrogram.cc
    ${MLEK_TEST_ROOT}/source/lib/mlek/common/MelSpectrogram.cc
    ${MLEK_TEST_ROOT}/source/lib/mlek/math/PlatformMath.cc)
target_include_directories(conformer_processing_tests PRIVATE ${MLEK_TEST_ROOT}/source/lib)
target_compile_features(conformer_processing_tests PRIVATE cxx_std_17)
target_compile_definitions(conformer_processing_tests PRIVATE MLEK_LOG_ENABLE)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT CMAKE_CROSSCOMPILING)
    target_compile_options(conformer_processing_tests PRIVATE -fsanitize=address,undefined)
    target_link_options(conformer_processing_tests PRIVATE -fsanitize=address,undefined)
endif()
add_test(NAME conformer-processing COMMAND conformer_processing_tests)

add_executable(asr_simple_stage_timings_tests AsrSimpleStageTimingsTests.cc)
target_compile_features(asr_simple_stage_timings_tests PRIVATE cxx_std_17)
target_include_directories(asr_simple_stage_timings_tests PRIVATE
    ${MLEK_TEST_ROOT}/source/app/use_case/asr/include)
add_test(NAME asr-simple-stage-timings COMMAND asr_simple_stage_timings_tests)

foreach(TIMING_MODE IN ITEMS normal simple detailed)
    set(CLASSIFIER_TEST_TARGET asr_classifier_${TIMING_MODE}_tests)
    add_executable(${CLASSIFIER_TEST_TARGET}
        AsrClassifierTimingTests.cc
        ${MLEK_TEST_ROOT}/source/lib/mlek/use_case/asr/AsrClassifier.cc
        ${MLEK_TEST_ROOT}/source/lib/mlek/common/Classifier.cc
        ${MLEK_TEST_ROOT}/source/lib/mlek/math/PlatformMath.cc)
    target_compile_features(${CLASSIFIER_TEST_TARGET} PRIVATE cxx_std_17)
    target_include_directories(${CLASSIFIER_TEST_TARGET} PRIVATE ${MLEK_TEST_ROOT}/source/lib)
    target_compile_definitions(${CLASSIFIER_TEST_TARGET} PRIVATE MLEK_LOG_ENABLE)
    if (TIMING_MODE STREQUAL "simple")
        target_compile_definitions(${CLASSIFIER_TEST_TARGET} PRIVATE ASR_SIMPLE_STAGE_PROFILING)
    elseif (TIMING_MODE STREQUAL "detailed")
        target_compile_definitions(${CLASSIFIER_TEST_TARGET} PRIVATE ASR_STAGE_PROFILING)
    endif()
    add_test(NAME asr-classifier-${TIMING_MODE} COMMAND ${CLASSIFIER_TEST_TARGET})
endforeach()

foreach(PROFILING_MODE IN ITEMS enabled disabled simple)
    set(SCOPE_TEST_TARGET processing_stage_scope_${PROFILING_MODE}_tests)
    add_executable(${SCOPE_TEST_TARGET} ProcessingStageScopeTests.cc)
    target_compile_features(${SCOPE_TEST_TARGET} PRIVATE cxx_std_17)
    target_include_directories(${SCOPE_TEST_TARGET} PRIVATE ${MLEK_TEST_ROOT}/source/lib)
    if (PROFILING_MODE STREQUAL "enabled")
        target_compile_definitions(${SCOPE_TEST_TARGET} PRIVATE ASR_STAGE_PROFILING)
    elseif (PROFILING_MODE STREQUAL "simple")
        target_compile_definitions(${SCOPE_TEST_TARGET} PRIVATE ASR_SIMPLE_STAGE_PROFILING)
    endif()
    add_test(NAME processing-stage-scope-${PROFILING_MODE} COMMAND ${SCOPE_TEST_TARGET})
endforeach()

if (EXISTS "${MLEK_TEST_ROOT}/dependencies/executorch/runtime/core/event_tracer.h")
    add_executable(conformer_op_profiler_tests ConformerOpProfilerTests.cc)
    target_compile_features(conformer_op_profiler_tests PRIVATE cxx_std_17)
    target_compile_definitions(conformer_op_profiler_tests PRIVATE
        ET_EVENT_TRACER_ENABLED C10_USING_CUSTOM_GENERATED_MACROS)
    target_include_directories(conformer_op_profiler_tests PRIVATE
        ${MLEK_TEST_ROOT}/dependencies
        ${MLEK_TEST_ROOT}/dependencies/executorch/runtime/core/portable_type/c10
        ${MLEK_TEST_ROOT}/source/app/use_case/asr/include)
    add_test(NAME conformer-op-profiler COMMAND conformer_op_profiler_tests)
endif()
