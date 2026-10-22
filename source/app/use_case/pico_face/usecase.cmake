# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
set(${use_case}_supports_${ML_FRAMEWORK} OFF)
if (NOT ML_FRAMEWORK STREQUAL "executorch")
    return()
endif()
if (USE_CASE_BUILD STREQUAL "all" AND NOT DEFINED pico_face_MODEL_PATH)
    message(STATUS "pico_face skipped: supply an exported PTE explicitly")
    return()
endif()
if (NOT TARGET_PLATFORM STREQUAL "mps4" OR NOT TARGET_SUBSYSTEM STREQUAL "sse-320" OR
    NOT ETHOS_U_NPU_ID STREQUAL "U85" OR
    NOT ETHOS_U_NPU_MEMORY_MODE MATCHES "^(Dedicated_Sram|Shared_Sram)$")
    message(FATAL_ERROR "pico_face requires SSE-320/U85 Dedicated_Sram or Shared_Sram")
endif()
set(${use_case}_supports_${ML_FRAMEWORK} ON)
list(APPEND ${use_case}_API_LIST pico_face)
USER_OPTION(pico_face_MODEL_PATH "Exported pico-face PTE" "" FILEPATH)
get_filename_component(_pico_model_dir "${pico_face_MODEL_PATH}" DIRECTORY)
USER_OPTION(pico_face_PARAMS_PATH "Matching exported conditioning table"
    "${_pico_model_dir}/PicoFaceParams.hpp" FILEPATH)
USER_OPTION(pico_face_ACTIVATION_BUF_SZ "DDR method pool capacity" 0x00400000 STRING)
USER_OPTION(pico_face_SEED "Deterministic generation seed" 3 STRING)
USER_OPTION(pico_face_STEPS "Euler steps: 1, 2, 4, 8" 4 STRING)
USER_OPTION(pico_face_CLASS_ID "Class 0..3, unconditional 4" 1 STRING)
USER_OPTION(pico_face_GUIDANCE "Classifier-free guidance scale" 4.0 STRING)
USER_OPTION(pico_face_SAVE_IMAGE "Save RGB via C stdio; enable only with semihosting" OFF BOOL)
USER_OPTION(pico_face_UART_IMAGE "Serve RGB frames over the FPGA UART after generation" OFF BOOL)
if (pico_face_UART_IMAGE AND (SEMIHOSTING_ENABLED OR NOT FPGA_PLATFORM_SSE_320))
    message(FATAL_ERROR "pico_face_UART_IMAGE requires FPGA with semihosting OFF")
endif()
if (pico_face_SAVE_IMAGE AND NOT SEMIHOSTING_ENABLED)
    message(FATAL_ERROR "pico_face_SAVE_IMAGE needs SEMIHOSTING_ENABLED")
endif()
if (NOT EXISTS "${pico_face_MODEL_PATH}" OR NOT EXISTS "${pico_face_PARAMS_PATH}")
    message(FATAL_ERROR "Export the pico-face model and matching parameter table first")
endif()
configure_file("${pico_face_PARAMS_PATH}" "${INC_GEN_DIR}/PicoFaceParams.hpp" COPYONLY)
generate_model_code(MODEL_PATH "${pico_face_MODEL_PATH}"
    DESTINATION ${SRC_GEN_DIR} NAMESPACE arm app pico_face)
generate_pte_ops_lib(MODEL_PATH "${pico_face_MODEL_PATH}"
    LIB_NAME pico_face_portable_ops_lib SELECT_OPS_LIST "")
if (TARGET pico_face_portable_ops_lib)
    list(APPEND ${use_case}_LINK_LIBS pico_face_portable_ops_lib)
endif()
set(${use_case}_COMPILE_DEFS
    "PICO_FACE_SEED=${pico_face_SEED}"
    "PICO_FACE_STEPS=${pico_face_STEPS}"
    "PICO_FACE_CLASS_ID=${pico_face_CLASS_ID}"
    "PICO_FACE_GUIDANCE=${pico_face_GUIDANCE}"
    "PICO_FACE_SAVE_IMAGE=$<BOOL:${pico_face_SAVE_IMAGE}>"
    "PICO_FACE_UART_IMAGE=$<BOOL:${pico_face_UART_IMAGE}>"
    "PICO_FACE_TEMP_BASE=${ML_FWK_TMP_MEM_BASE}"
    "PICO_FACE_TEMP_SIZE=${ML_FWK_TMP_MEM_SIZE}")
