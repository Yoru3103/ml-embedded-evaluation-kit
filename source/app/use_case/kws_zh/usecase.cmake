# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

set(${use_case}_ML_FRAMEWORK "${ML_FRAMEWORK_EXECUTORCH}")
if (NOT ${ML_FRAMEWORK} IN_LIST ${use_case}_ML_FRAMEWORK)
    set(${use_case}_supports_${ML_FRAMEWORK} OFF)
    return()
endif()
set(_model_directory "pte_${ETHOS_U_NPU_MEMORY_MODE}")
if (ETHOS_U_NPU_MEMORY_MODE STREQUAL "Dedicated_Sram")
    math(EXPR _cache_size_kib "${ETHOS_U_NPU_CACHE_SIZE} / 1024")
    string(APPEND _model_directory "_${_cache_size_kib}k")
endif()
set(DEFAULT_MODEL_PATH
    ${MLEK_ROOT}/resources_downloaded/kws_zh/${_model_directory}/wekws_dstcn_u85.pte)
if ("${USE_CASE_BUILD}" STREQUAL "all" AND
    NOT DEFINED ${use_case}_MODEL_PATH AND NOT EXISTS "${DEFAULT_MODEL_PATH}")
    message(STATUS "kws_zh skipped: supply the exported WeKWS PTE model")
    set(${use_case}_supports_${ML_FRAMEWORK} OFF)
    return()
endif()
set(${use_case}_supports_${ML_FRAMEWORK} ON)
USER_OPTION(${use_case}_INPUT_MODE "Chinese KWS input: STATIC, VSI or UART" STATIC STRING)
set_property(CACHE ${use_case}_INPUT_MODE PROPERTY STRINGS STATIC VSI UART)
if (NOT ${use_case}_INPUT_MODE MATCHES "^(STATIC|VSI|UART)$")
    message(FATAL_ERROR "kws_zh_INPUT_MODE must be STATIC, VSI or UART")
endif()
if (${use_case}_INPUT_MODE STREQUAL "VSI")
    include(${MLEK_ROOT}/scripts/cmake/fvp_audio_demo.cmake)
    mlek_add_fvp_audio_demo(${use_case})
    USER_OPTION(${use_case}_STREAM_HOP_SAMPLES "Live KWS inference hop in PCM samples"
        16000 STRING)
    math(EXPR _stream_remainder "${${use_case}_STREAM_HOP_SAMPLES} % 160")
    if (${${use_case}_STREAM_HOP_SAMPLES} LESS 160 OR
        ${${use_case}_STREAM_HOP_SAMPLES} GREATER 31920 OR NOT _stream_remainder EQUAL 0)
        message(FATAL_ERROR "KWS stream hop must be 160..31920 and a multiple of 160")
    endif()
    add_library(kws_zh_vsi_mode INTERFACE)
    target_compile_definitions(kws_zh_vsi_mode INTERFACE KWS_ZH_INPUT_VSI
        KWS_ZH_STREAM_HOP_SAMPLES=${${use_case}_STREAM_HOP_SAMPLES})
    list(APPEND ${use_case}_LINK_LIBS kws_zh_vsi_mode)
endif()
list(APPEND ${use_case}_API_LIST "kws_zh")
if (TARGET_PLATFORM STREQUAL "mps4" AND TARGET_SUBSYSTEM STREQUAL "sse-320" AND
    ETHOS_U_NPU_MEMORY_MODE STREQUAL "Dedicated_Sram" AND DEFINED ML_FWK_TMP_MEM_BASE)
    math(EXPR _temporary_base "${ML_FWK_TMP_MEM_BASE}")
    if (_temporary_base GREATER_EQUAL 0x31000000 AND _temporary_base LESS 0x31400000)
        message(FATAL_ERROR
            "kws_zh Dedicated scratch uses AXI EXT: set ML_FWK_TMP_MEM_BASE to DDR, e.g. 0x72000000")
    endif()
endif()

USER_OPTION(${use_case}_MODEL_PATH "Fixed-window WeKWS DS-TCN PTE."
    ${DEFAULT_MODEL_PATH} FILEPATH)
set_input_file_path_user_option(".wav" ${use_case})
USER_OPTION(${use_case}_LABELS_TXT_FILE "Labels in the checkpoint output order."
    ${MLEK_ROOT}/resources/kws_zh/labels/labels.txt FILEPATH)
USER_OPTION(${use_case}_ACTIVATION_BUF_SZ "ExecuTorch planned memory pool." 0x00100000 STRING)
USER_OPTION(${use_case}_SCORE_THRESHOLD "Independent sigmoid detection threshold." 0.5 STRING)
USER_OPTION(${use_case}_MINIMUM_FRAMES "Consecutive frames above threshold." 3 STRING)
USER_OPTION(${use_case}_COOLDOWN_MS "Per-keyword cooldown in milliseconds." 1000 STRING)
USER_OPTION(${use_case}_WINDOW_STRIDE "Audio samples between fixed windows." 16000 STRING)
USER_OPTION(${use_case}_DUMP_OUTPUT "Print per-frame sigmoid scores for validation." OFF BOOL)
if (${${use_case}_WINDOW_STRIDE} LESS 160 OR ${${use_case}_WINDOW_STRIDE} GREATER 31920)
    message(FATAL_ERROR "kws_zh_WINDOW_STRIDE must be between 160 and 31920 samples")
endif()
math(EXPR _stride_remainder "${${use_case}_WINDOW_STRIDE} % 160")
if (NOT _stride_remainder EQUAL 0)
    message(FATAL_ERROR "kws_zh_WINDOW_STRIDE must align to the 160-sample feature hop")
endif()
if (${${use_case}_MINIMUM_FRAMES} LESS 1 OR ${${use_case}_COOLDOWN_MS} LESS 0)
    message(FATAL_ERROR "kws_zh requires positive MINIMUM_FRAMES and nonnegative COOLDOWN_MS")
endif()

generate_audio_code(${${use_case}_FILE_PATH} ${SAMPLES_GEN_DIR} 16000 ON 0 0 kaiser_best 31920)
generate_labels_code(
    INPUT ${${use_case}_LABELS_TXT_FILE}
    DESTINATION_SRC ${SRC_GEN_DIR}
    DESTINATION_HDR ${INC_GEN_DIR}
    OUTPUT_FILENAME Labels)
generate_model_code(
    MODEL_PATH ${${use_case}_MODEL_PATH}
    DESTINATION ${SRC_GEN_DIR}
    NAMESPACE "arm" "app" "kws_zh")
if (COMMAND generate_pte_ops_lib)
    generate_pte_ops_lib(
        MODEL_PATH "${${use_case}_MODEL_PATH}"
        LIB_NAME "${use_case}_portable_ops_lib"
        SELECT_OPS_LIST "")
    if (TARGET ${use_case}_portable_ops_lib)
        list(APPEND ${use_case}_LINK_LIBS ${use_case}_portable_ops_lib)
    endif()
endif()
set(${use_case}_COMPILE_DEFS
    "KWS_ZH_SCORE_THRESHOLD=${${use_case}_SCORE_THRESHOLD}F"
    "KWS_ZH_MINIMUM_FRAMES=${${use_case}_MINIMUM_FRAMES}"
    "KWS_ZH_COOLDOWN_MS=${${use_case}_COOLDOWN_MS}"
    "KWS_ZH_DUMP_OUTPUT=$<BOOL:${${use_case}_DUMP_OUTPUT}>"
    "KWS_ZH_WINDOW_STRIDE=${${use_case}_WINDOW_STRIDE}")

if (${use_case}_INPUT_MODE STREQUAL "UART")
    if (NOT TARGET_PLATFORM STREQUAL "mps4" OR NOT TARGET_SUBSYSTEM STREQUAL "sse-320" OR SEMIHOSTING_ENABLED)
        message(FATAL_ERROR "KWS file UART requires SSE-320 and UART stdout")
    endif()
    list(APPEND ${use_case}_COMPILE_DEFS KWS_ZH_INPUT_UART)
endif()
