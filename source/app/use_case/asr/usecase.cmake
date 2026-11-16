#----------------------------------------------------------------------------
#  SPDX-FileCopyrightText: Copyright 2021, 2024-2026 Arm Limited and/or its
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

# Specify the ML frameworks the use case supports
set(${use_case}_ML_FRAMEWORK
    "${ML_FRAMEWORK_TFLM}"
    "${ML_FRAMEWORK_EXECUTORCH}")
if (NOT ${ML_FRAMEWORK} IN_LIST ${use_case}_ML_FRAMEWORK)
    set(${use_case}_supports_${ML_FRAMEWORK} OFF)
    return()
endif ()

# Default Conformer model is not supported on all NPUs yet.
# TODO: Remove this check once this limitation is resolved.
if (DEFINED ETHOS_U_NPU_ID)
    if ("${ML_FRAMEWORK}" STREQUAL "${ML_FRAMEWORK_EXECUTORCH}"
        AND ${ETHOS_U_NPU_ID} STREQUAL "U55"
        AND NOT DEFINED ${use_case}_MODEL_PATH)
        message(FATAL_ERROR "Arm Ethos-U55 NPU is not supported with Conformer model yet."
                            "Build this example for a platform with Arm Ethos-U85 NPU instead.")
        return()
    endif()
endif()

set(${use_case}_supports_${ML_FRAMEWORK} ON)

USER_OPTION(${use_case}_INPUT_MODE "ASR input source: STATIC, UART or VSI" STATIC STRING)
set_property(CACHE ${use_case}_INPUT_MODE PROPERTY STRINGS STATIC UART VSI)
if (NOT ${use_case}_INPUT_MODE MATCHES "^(STATIC|UART|VSI)$")
    message(FATAL_ERROR "asr_INPUT_MODE must be STATIC, UART or VSI")
endif()
if (${use_case}_INPUT_MODE STREQUAL "VSI")
    if (NOT "${ML_FRAMEWORK}" STREQUAL "${ML_FRAMEWORK_EXECUTORCH}")
        message(FATAL_ERROR "ASR VSI requires ExecuTorch Conformer")
    endif()
    include(${MLEK_ROOT}/scripts/cmake/fvp_audio_demo.cmake)
    mlek_add_fvp_audio_demo(${use_case})
    USER_OPTION(${use_case}_VSI_VAD_RMS "VSI speech RMS threshold in 16-bit PCM units"
        800 STRING)
    USER_OPTION(${use_case}_VSI_SILENCE_MS "VSI sentence-end silence in milliseconds"
        1000 STRING)
    USER_OPTION(${use_case}_VSI_PRE_ROLL_MS "VSI audio retained before speech in milliseconds"
        600 STRING)
    if (NOT "${${use_case}_VSI_PRE_ROLL_MS}" MATCHES "^[1-9][0-9]*$" OR
        ${use_case}_VSI_PRE_ROLL_MS LESS 200 OR ${use_case}_VSI_PRE_ROLL_MS GREATER 2000)
        message(FATAL_ERROR "asr_VSI_PRE_ROLL_MS must be an integer from 200 to 2000")
    endif()
    if (NOT "${${use_case}_VSI_VAD_RMS}" MATCHES "^[1-9][0-9]*$" OR
        ${use_case}_VSI_VAD_RMS GREATER_EQUAL 32768)
        message(FATAL_ERROR "asr_VSI_VAD_RMS must be an integer from 1 to 32767")
    endif()
    if (NOT "${${use_case}_VSI_SILENCE_MS}" MATCHES "^[1-9][0-9]*$" OR
        ${use_case}_VSI_SILENCE_MS LESS 20 OR ${use_case}_VSI_SILENCE_MS GREATER 5000)
        message(FATAL_ERROR "asr_VSI_SILENCE_MS must be an integer from 20 to 5000")
    endif()
    add_library(asr_vsi_mode INTERFACE)
    target_compile_definitions(asr_vsi_mode INTERFACE ASR_INPUT_VSI
        ASR_VSI_VAD_RMS=${${use_case}_VSI_VAD_RMS}
        ASR_VSI_PRE_ROLL_MS=${${use_case}_VSI_PRE_ROLL_MS}
        ASR_VSI_SILENCE_MS=${${use_case}_VSI_SILENCE_MS})
    list(APPEND ${use_case}_LINK_LIBS asr_vsi_mode)
endif()
if (${use_case}_INPUT_MODE STREQUAL "UART")
    if (NOT "${ML_FRAMEWORK}" STREQUAL "${ML_FRAMEWORK_EXECUTORCH}" OR
        NOT TARGET_PLATFORM STREQUAL "mps4" OR NOT TARGET_SUBSYSTEM STREQUAL "sse-320" OR
        SEMIHOSTING_ENABLED OR NOT ETHOS_U_NPU_ENABLED OR
        NOT ETHOS_U_NPU_MEMORY_MODE STREQUAL "Dedicated_Sram" OR
        NOT ETHOS_U_NPU_CACHE_SIZE EQUAL 262144 OR NOT ASR_ET_STAGE_PROFILING)
        message(FATAL_ERROR "ASR UART requires MPS4 SSE-320, ExecuTorch, UART stdout, Dedicated 256 KiB and ASR_ET_STAGE_PROFILING")
    endif()
    add_library(asr_uart_mode INTERFACE)
    target_compile_definitions(asr_uart_mode INTERFACE ASR_INPUT_UART
        ASR_UART_FPGA=$<BOOL:${FPGA_PLATFORM_SSE_320}>
        $<$<BOOL:${MPS4_DISPLAY_ENABLED}>:ASR_UART_LCD>)
    list(APPEND ${use_case}_LINK_LIBS asr_uart_mode)
endif()

if (ASR_SIMPLE_STAGE_PROFILING)
    if (NOT "${ML_FRAMEWORK}" STREQUAL "${ML_FRAMEWORK_TFLM}" OR
        NOT ETHOS_U_NPU_ENABLED OR NOT TARGET_PLATFORM STREQUAL "mps4" OR
        NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(FATAL_ERROR "Wav2letter simple timing requires MPS4 TFLM NPU and linker wrapping")
    endif()
    add_library(asr_simple_stage_hooks INTERFACE)
    target_link_options(asr_simple_stage_hooks INTERFACE "LINKER:--wrap=ethosu_invoke_v3")
    list(APPEND ${use_case}_LINK_LIBS asr_simple_stage_hooks)
endif()

if (ASR_ET_STAGE_PROFILING AND "${ML_FRAMEWORK}" STREQUAL "${ML_FRAMEWORK_EXECUTORCH}")
    if (NOT ETHOS_U_NPU_ENABLED OR NOT TARGET_PLATFORM STREQUAL "mps4" OR
        NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(FATAL_ERROR "Conformer stage timing requires MPS4 NPU and GNU/Clang linker wrapping")
    endif()
    add_library(asr_et_stage_hooks INTERFACE)
    target_compile_definitions(asr_et_stage_hooks INTERFACE ASR_ET_STAGE_PROFILING)
    target_link_options(asr_et_stage_hooks INTERFACE "LINKER:--wrap=ethosu_invoke_v3")
    list(APPEND ${use_case}_LINK_LIBS asr_et_stage_hooks)
endif()

# Append the API to use for this use case
list(APPEND ${use_case}_API_LIST "asr")

set_input_file_path_user_option(".wav" ${use_case})

if ("${ML_FRAMEWORK}" STREQUAL "${ML_FRAMEWORK_EXECUTORCH}")
    USER_OPTION(${use_case}_LABELS_TXT_FILE "Labels' txt file for the chosen model."
        ${MLEK_ROOT}/resources/${use_case}/labels/librispeech_sp.pieces
        FILEPATH)
else ()
    USER_OPTION(${use_case}_LABELS_TXT_FILE "Labels' txt file for the chosen model."
        ${MLEK_ROOT}/resources/${use_case}/labels/labels_wav2letter.txt
        FILEPATH)
endif ()

USER_OPTION(${use_case}_AUDIO_RATE "Specify the target sampling rate. Default is 16000."
    16000
    STRING)

USER_OPTION(${use_case}_AUDIO_MONO "Specify if the audio needs to be converted to mono. Default is ON."
    ON
    BOOL)

USER_OPTION(${use_case}_AUDIO_OFFSET "Specify the offset to start reading after this time (in seconds). Default is 0."
    0
    STRING)

USER_OPTION(${use_case}_AUDIO_DURATION "Specify the audio duration to load (in seconds). If set to 0 the entire audio will be processed."
    0
    STRING)

USER_OPTION(${use_case}_AUDIO_RES_TYPE "Specify re-sampling algorithm to use. By default is 'kaiser_best'."
    kaiser_best
    STRING)

USER_OPTION(${use_case}_AUDIO_MIN_SAMPLES "Specify the minimum number of samples to use. By default is 16000, if the audio is shorter will be automatically padded."
    16000
    STRING)

USER_OPTION(${use_case}_MODEL_SCORE_THRESHOLD "Specify the score threshold [0.0, 1.0) that must be applied to the inference results for a label to be deemed valid."
    0.5
    STRING)

# Generate input files
generate_audio_code(${${use_case}_FILE_PATH} ${SAMPLES_GEN_DIR}
    ${${use_case}_AUDIO_RATE}
    ${${use_case}_AUDIO_MONO}
    ${${use_case}_AUDIO_OFFSET}
    ${${use_case}_AUDIO_DURATION}
    ${${use_case}_AUDIO_RES_TYPE}
    ${${use_case}_AUDIO_MIN_SAMPLES})

# Generate labels file
set(${use_case}_LABELS_CPP_FILE Labels)
generate_labels_code(
    INPUT           "${${use_case}_LABELS_TXT_FILE}"
    DESTINATION_SRC ${SRC_GEN_DIR}
    DESTINATION_HDR ${INC_GEN_DIR}
    OUTPUT_FILENAME "${${use_case}_LABELS_CPP_FILE}"
)

if ("${ML_FRAMEWORK}" STREQUAL "${ML_FRAMEWORK_TFLM}")
    USER_OPTION(${use_case}_ACTIVATION_BUF_SZ "Activation buffer size for the chosen model"
        0x00200000
        STRING)

    if (ETHOS_U_NPU_ENABLED)
        set(DEFAULT_MODEL_PATH      ${DEFAULT_MODEL_DIR}/wav2letter_pruned_int8_vela_${ETHOS_U_NPU_CONFIG_ID}_${ETHOS_U_NPU_MEMORY_MODE}.tflite)
    else()
        set(DEFAULT_MODEL_PATH      ${DEFAULT_MODEL_DIR}/wav2letter_pruned_int8.tflite)
    endif()

    set(EXTRA_MODEL_CODE
        "/* Model parameters for ${use_case} */"
        "extern const int   g_FrameLength    = 512"
        "extern const int   g_FrameStride    = 160"
        "extern const int   g_ctxLen         =  98"
        "extern const float g_ScoreThreshold = ${${use_case}_MODEL_SCORE_THRESHOLD}"
        )
elseif("${ML_FRAMEWORK}" STREQUAL "${ML_FRAMEWORK_EXECUTORCH}")

    if (ETHOS_U_NPU_ENABLED)
        string(TOLOWER ${ETHOSU_TARGET_NPU_CONFIG} _NPU_CFG_ID)
        set(DEFAULT_MODEL_PATH      ${DEFAULT_MODEL_DIR}/conformer_fp32_cln_wer_6_47_arm_delegate_${_NPU_CFG_ID}_${ETHOS_U_NPU_MEMORY_MODE}.pte)
        set(DEFAULT_ACT_BUF_SZ      0x00200000) # 2 MiB
    else()
        set(DEFAULT_MODEL_PATH      ${DEFAULT_MODEL_DIR}/conformer_fp32_cln_wer_6_47_arm_TOSA-1.0+FP.pte)
        set(DEFAULT_ACT_BUF_SZ      0x03000000) # 48 MiB
    endif()

    USER_OPTION(${use_case}_ACTIVATION_BUF_SZ "Activation buffer size for the chosen model"
        ${DEFAULT_ACT_BUF_SZ}
        STRING)

    set(EXTRA_MODEL_CODE
        "/* Model parameters for ${use_case} */"
        "extern const int   g_melSpecWindowSize = 512"
        "extern const int   g_melSpecHopSize    = 160"
        "extern const int   g_chunkSize         = 700"
        )
endif()

USER_OPTION(${use_case}_MODEL_PATH "NN models file to be used in the evaluation application. Model files must be in tflite or pte format."
    ${DEFAULT_MODEL_PATH}
    FILEPATH
    )

if (COMMAND generate_pte_ops_lib)
    generate_pte_ops_lib(
        MODEL_PATH      "${${use_case}_MODEL_PATH}"     # Path to the model PTE
        LIB_NAME        "${use_case}_portable_ops_lib"  # Library target name
        SELECT_OPS_LIST "")                             # Always included ops list

    # If the target is generated, request it to be linked for this use case.
    if (TARGET ${use_case}_portable_ops_lib)
        list(APPEND ${use_case}_LINK_LIBS ${use_case}_portable_ops_lib)
    endif()
endif()

# Generate model file
generate_model_code(
    MODEL_PATH ${${use_case}_MODEL_PATH}
    DESTINATION ${SRC_GEN_DIR}
    EXPRESSIONS ${EXTRA_MODEL_CODE}
    NAMESPACE   "arm" "app" "asr")
