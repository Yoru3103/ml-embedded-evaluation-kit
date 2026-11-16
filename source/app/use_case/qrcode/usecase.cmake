# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

set(${use_case}_ML_FRAMEWORK "${ML_FRAMEWORK_TFLM}" "${ML_FRAMEWORK_EXECUTORCH}")
if (NOT ${ML_FRAMEWORK} IN_LIST ${use_case}_ML_FRAMEWORK)
    set(${use_case}_supports_${ML_FRAMEWORK} OFF)
    return()
endif()
set(${use_case}_supports_${ML_FRAMEWORK} ON)
USER_OPTION(${use_case}_INPUT_MODE "QR input source: STATIC, SEMIHOST or UART" STATIC STRING)
set_property(CACHE ${use_case}_INPUT_MODE PROPERTY STRINGS STATIC SEMIHOST UART)
if (NOT ${use_case}_INPUT_MODE MATCHES "^(STATIC|SEMIHOST|UART)$")
    message(FATAL_ERROR "qrcode_INPUT_MODE must be STATIC, SEMIHOST or UART")
endif()
if (${use_case}_INPUT_MODE STREQUAL "SEMIHOST" AND
    (NOT TARGET_PLATFORM STREQUAL "mps4" OR NOT TARGET_SUBSYSTEM STREQUAL "sse-320" OR
     FPGA_PLATFORM_SSE_320 OR NOT SEMIHOSTING_ENABLED OR NOT MPS4_DISPLAY_ENABLED))
    message(FATAL_ERROR "QR semihost demo requires SSE-320 FVP, semihosting and LCD")
endif()

if ("${ML_FRAMEWORK}" STREQUAL "${ML_FRAMEWORK_TFLM}")
    set(DEFAULT_MODEL_PATH
        ${RESOURCES_PATH}/qrcode/tflm_${ETHOS_U_NPU_MEMORY_MODE}/best_int8_vela.tflite)
else()
    set(DEFAULT_MODEL_PATH
        ${RESOURCES_PATH}/qrcode/pte_${ETHOS_U_NPU_MEMORY_MODE}/best.pte)
endif()

# This custom model is not installed by the default resource setup.
if ("${USE_CASE_BUILD}" STREQUAL "all" AND
    NOT DEFINED ${use_case}_MODEL_PATH AND NOT EXISTS "${DEFAULT_MODEL_PATH}")
    message(STATUS "qrcode skipped: supply qrcode_MODEL_PATH to enable the grayscale model")
    set(${use_case}_supports_${ML_FRAMEWORK} OFF)
    return()
endif()

# Reuse the angle-aware DFL decoder and LCD box drawing for grayscale,
# four-output QR-code models at the configured input width and height.
list(APPEND ${use_case}_API_LIST "yolov8_detection")
add_library(qrcode_drawing STATIC
    ${CMAKE_CURRENT_LIST_DIR}/../yolov8_detection/src/DetectionDrawing.cc)
target_include_directories(qrcode_drawing PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../yolov8_detection/include)
target_link_libraries(qrcode_drawing PUBLIC hal yolov8_detection_api)
list(APPEND ${use_case}_LINK_LIBS qrcode_drawing)

USER_OPTION(${use_case}_MAX_DETECTIONS "Maximum detections after NMS." 20 STRING)
USER_OPTION(${use_case}_SCORE_THRESHOLD "Class confidence threshold." 0.25 STRING)
USER_OPTION(${use_case}_NMS_THRESHOLD "Rotated NMS IoU threshold." 0.45 STRING)
USER_OPTION(${use_case}_DISPLAY_DOWNSCALE "LCD image downscale factor." 2 STRING)
USER_OPTION(${use_case}_ACTIVATION_BUF_SZ "Tensor arena size." 0x00400000 STRING)
USER_OPTION(${use_case}_ET_METHOD_MEMORY_IN_DDR
    "Place ExecuTorch method memory in DDR." OFF BOOL)

USER_OPTION(${use_case}_MODEL_PATH "Grayscale QR-code TFLite or PTE model."
    ${DEFAULT_MODEL_PATH} FILEPATH)
USER_OPTION(${use_case}_FILE_PATH "Grayscale BMP input image or directory."
    ${CMAKE_SOURCE_DIR}/resources/qrcode/smaples PATH_OR_FILE)
USER_OPTION(${use_case}_LABELS_FILE "Fifteen QR-code class names."
    ${CMAKE_SOURCE_DIR}/resources/qrcode/labels.txt FILEPATH)

if (NOT EXISTS "${${use_case}_LABELS_FILE}")
    message(FATAL_ERROR "QR-code labels file not found: ${${use_case}_LABELS_FILE}")
endif()
file(STRINGS "${${use_case}_LABELS_FILE}" QRCODE_LABELS)
list(LENGTH QRCODE_LABELS QRCODE_LABEL_COUNT)
if (NOT QRCODE_LABEL_COUNT EQUAL 15)
    message(FATAL_ERROR "QR-code model requires 15 labels; found ${QRCODE_LABEL_COUNT}")
endif()

USER_OPTION(${use_case}_IMAGE_WIDTH "QR-code input width." 256 STRING)
USER_OPTION(${use_case}_IMAGE_HEIGHT "QR-code input height." 256 STRING)
if (${use_case}_INPUT_MODE STREQUAL "SEMIHOST")
    add_library(qrcode_semihost_mode INTERFACE)
    target_compile_definitions(qrcode_semihost_mode INTERFACE QRCODE_INPUT_SEMIHOST
        QRCODE_INPUT_WIDTH=${${use_case}_IMAGE_WIDTH}
        QRCODE_INPUT_HEIGHT=${${use_case}_IMAGE_HEIGHT})
    list(APPEND ${use_case}_LINK_LIBS qrcode_semihost_mode)
endif()
USER_OPTION(${use_case}_IMAGE_RESIZE_MODE "QR-code image resize mode: crop or letterbox."
    letterbox STRING)

generate_images_code(
    "${${use_case}_FILE_PATH}"
    ${SAMPLES_GEN_DIR}
    ${${use_case}_IMAGE_WIDTH}
    HEIGHT ${${use_case}_IMAGE_HEIGHT}
    CHANNELS 1
    RESIZE_MODE ${${use_case}_IMAGE_RESIZE_MODE})
generate_labels_code(
    INPUT "${${use_case}_LABELS_FILE}"
    DESTINATION_SRC ${SRC_GEN_DIR}
    DESTINATION_HDR ${INC_GEN_DIR}
    OUTPUT_FILENAME Labels)
generate_model_code(
    MODEL_PATH ${${use_case}_MODEL_PATH}
    DESTINATION ${SRC_GEN_DIR}
    NAMESPACE "arm" "app" "qrcode")

if (COMMAND generate_pte_ops_lib AND
    "${ML_FRAMEWORK}" STREQUAL "${ML_FRAMEWORK_EXECUTORCH}")
    generate_pte_ops_lib(
        MODEL_PATH "${${use_case}_MODEL_PATH}"
        LIB_NAME "${use_case}_portable_ops_lib"
        SELECT_OPS_LIST "")
    if (TARGET ${use_case}_portable_ops_lib)
        list(APPEND ${use_case}_LINK_LIBS ${use_case}_portable_ops_lib)
    endif()
endif()

set(${use_case}_COMPILE_DEFS
    "QRCODE_ET_METHOD_MEMORY_IN_DDR=$<BOOL:${${use_case}_ET_METHOD_MEMORY_IN_DDR}>"
    "QRCODE_MAX_DETECTIONS=${${use_case}_MAX_DETECTIONS}"
    "QRCODE_SCORE_THRESHOLD=${${use_case}_SCORE_THRESHOLD}F"
    "QRCODE_NMS_THRESHOLD=${${use_case}_NMS_THRESHOLD}F"
    "QRCODE_DISPLAY_DOWNSCALE=${${use_case}_DISPLAY_DOWNSCALE}")

if (${use_case}_INPUT_MODE STREQUAL "UART")
    if (NOT TARGET_PLATFORM STREQUAL "mps4" OR NOT TARGET_SUBSYSTEM STREQUAL "sse-320" OR SEMIHOSTING_ENABLED)
        message(FATAL_ERROR "File UART input requires SSE-320 and UART stdout")
    endif()
    add_library(qrcode_uart_file_mode INTERFACE)
    target_compile_definitions(qrcode_uart_file_mode INTERFACE QRCODE_INPUT_UART
        QRCODE_INPUT_WIDTH=${${use_case}_IMAGE_WIDTH}
        QRCODE_INPUT_HEIGHT=${${use_case}_IMAGE_HEIGHT})
    list(APPEND ${use_case}_LINK_LIBS qrcode_uart_file_mode)
endif()
