# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

# Share pose processing between the two runtime integrations.
set(${use_case}_ML_FRAMEWORK "${ML_FRAMEWORK_TFLM};${ML_FRAMEWORK_EXECUTORCH}")
if (NOT ${ML_FRAMEWORK} IN_LIST ${use_case}_ML_FRAMEWORK)
    set(${use_case}_supports_${ML_FRAMEWORK} OFF)
    return()
endif()

set(${use_case}_supports_${ML_FRAMEWORK} ON)
list(APPEND ${use_case}_API_LIST "yolov8_pose")

USER_OPTION(
    ${use_case}_MODEL_IN_EXT_FLASH
    "Run model from external flash."
    OFF
    BOOL)

USER_OPTION(
    ${use_case}_IMAGE_SIZE
    "YOLOv8 pose input image width and height."
    224
    STRING)

USER_OPTION(
    ${use_case}_NUM_CLASSES
    "Number of YOLOv8 pose classes."
    1
    STRING)

USER_OPTION(
    ${use_case}_MAX_DETECTIONS
    "Maximum number of poses after NMS."
    20
    STRING)

USER_OPTION(
    ${use_case}_SCORE_THRESHOLD
    "Pose detection score threshold."
    0.25
    STRING)

USER_OPTION(
    ${use_case}_NMS_THRESHOLD
    "Pose NMS IoU threshold."
    0.45
    STRING)

USER_OPTION(
    ${use_case}_KEYPOINT_THRESHOLD
    "Minimum keypoint confidence for display and logging."
    0.25
    STRING)

if (ML_FRAMEWORK STREQUAL "${ML_FRAMEWORK_TFLM}")
    set(YOLOV8_POSE_DEFAULT_NORMALIZED ON)
    set(YOLOV8_POSE_DEFAULT_TOPOLOGY PACKED)
else()
    set(YOLOV8_POSE_DEFAULT_NORMALIZED OFF)
    set(YOLOV8_POSE_DEFAULT_TOPOLOGY SPLIT)
endif()

USER_OPTION(
    ${use_case}_COORDINATES_NORMALIZED
    "Model box and keypoint coordinates are normalized to [0, 1]."
    ${YOLOV8_POSE_DEFAULT_NORMALIZED}
    BOOL)

USER_OPTION(
    ${use_case}_OUTPUT_TOPOLOGY
    "Expected model output topology: AUTO, PACKED, or SPLIT."
    ${YOLOV8_POSE_DEFAULT_TOPOLOGY}
    STRING)

set(YOLOV8_POSE_OUTPUT_TOPOLOGY "${${use_case}_OUTPUT_TOPOLOGY}")
string(TOUPPER "${YOLOV8_POSE_OUTPUT_TOPOLOGY}" YOLOV8_POSE_OUTPUT_TOPOLOGY)
set(YOLOV8_POSE_VALID_OUTPUT_TOPOLOGIES AUTO PACKED SPLIT)
if (NOT YOLOV8_POSE_OUTPUT_TOPOLOGY IN_LIST YOLOV8_POSE_VALID_OUTPUT_TOPOLOGIES)
    message(FATAL_ERROR
        "${use_case}_OUTPUT_TOPOLOGY must be AUTO, PACKED, or SPLIT; got "
        "${${use_case}_OUTPUT_TOPOLOGY}")
endif()

if (YOLOV8_POSE_OUTPUT_TOPOLOGY STREQUAL "PACKED")
    set(YOLOV8_POSE_EXPECTED_NUM_OUTPUTS 1)
elseif (YOLOV8_POSE_OUTPUT_TOPOLOGY STREQUAL "SPLIT")
    set(YOLOV8_POSE_EXPECTED_NUM_OUTPUTS 4)
else()
    set(YOLOV8_POSE_EXPECTED_NUM_OUTPUTS 0)
endif()

USER_OPTION(
    ${use_case}_DISPLAY_DOWNSCALE
    "LCD image downscale factor."
    2
    STRING)

USER_OPTION(
    ${use_case}_PMU_WARMUP_RUNS
    "Number of unprofiled warm-up inferences before PMU sampling."
    0
    STRING)

USER_OPTION(
    ${use_case}_PMU_MEASURED_RUNS
    "Number of profiled repeated inferences to aggregate."
    1
    STRING)

if (NOT "${${use_case}_PMU_WARMUP_RUNS}" MATCHES "^[0-9]+$")
    message(FATAL_ERROR "${use_case}_PMU_WARMUP_RUNS must be a non-negative integer")
endif()
if (NOT "${${use_case}_PMU_MEASURED_RUNS}" MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "${use_case}_PMU_MEASURED_RUNS must be a positive integer")
endif()

USER_OPTION(
    ${use_case}_ACTIVATION_BUF_SZ
    "Tensor arena size."
    0x00200000
    STRING)

if (ML_FRAMEWORK STREQUAL "${ML_FRAMEWORK_TFLM}")
    set(DEFAULT_MODEL_PATH
        ${CMAKE_SOURCE_DIR}/resources/yolov8_pose/model_int8_vela_u85_1024_packed_dedicated_128k.tflite)
elseif (FPGA_PLATFORM_SSE_320)
    set(DEFAULT_MODEL_PATH
        ${CMAKE_SOURCE_DIR}/resources/yolov8_pose/yolo_hand_pose_dedicated_128k_MPS4_ethos_u85_1024.pte)
else()
    set(DEFAULT_MODEL_PATH
        ${CMAKE_SOURCE_DIR}/resources/yolov8_pose/yolo_hand_pose_dedicated_128k_ethos_u85_1024.pte)
endif()

USER_OPTION(
    ${use_case}_MODEL_PATH
    "YOLOv8 pose model matching the selected ML framework."
    ${DEFAULT_MODEL_PATH}
    FILEPATH)

USER_OPTION(
    ${use_case}_FILE_PATH
    "Directory containing input images, or a path to one input image."
    ${CMAKE_SOURCE_DIR}/resources/yolov8_pose/samples
    PATH_OR_FILE)

USER_OPTION(
    ${use_case}_LABELS_YAML_FILE
    "Ultralytics dataset YAML containing the class names."
    ${CMAKE_SOURCE_DIR}/resources/yolov8_pose/hand_pose.yaml
    FILEPATH)

if (NOT EXISTS "${${use_case}_LABELS_YAML_FILE}")
    message(FATAL_ERROR
        "YOLOv8 pose labels YAML not found: ${${use_case}_LABELS_YAML_FILE}")
endif()

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${${use_case}_LABELS_YAML_FILE}")

file(STRINGS "${${use_case}_LABELS_YAML_FILE}" YOLOV8_POSE_LABEL_LINES
    REGEX "^[ \t]*[0-9]+:[ \t]+.+$")

list(LENGTH YOLOV8_POSE_LABEL_LINES YOLOV8_POSE_LABEL_COUNT)
if (NOT YOLOV8_POSE_LABEL_COUNT EQUAL ${${use_case}_NUM_CLASSES})
    message(FATAL_ERROR
        "Expected ${${use_case}_NUM_CLASSES} labels in "
        "${${use_case}_LABELS_YAML_FILE}, found ${YOLOV8_POSE_LABEL_COUNT}")
endif()

set(YOLOV8_POSE_LABELS_TXT_FILE "${SRC_GEN_DIR}/yolov8_pose_labels.txt")
file(WRITE "${YOLOV8_POSE_LABELS_TXT_FILE}" "")

set(YOLOV8_POSE_EXPECTED_CLASS_ID 0)
foreach(YOLOV8_POSE_LABEL_LINE IN LISTS YOLOV8_POSE_LABEL_LINES)
    string(REGEX MATCH
        "^[ \t]*([0-9]+):[ \t]*(.+)$"
        YOLOV8_POSE_LABEL_MATCH
        "${YOLOV8_POSE_LABEL_LINE}")
    set(YOLOV8_POSE_CLASS_ID "${CMAKE_MATCH_1}")
    set(YOLOV8_POSE_CLASS_NAME "${CMAKE_MATCH_2}")
    string(STRIP "${YOLOV8_POSE_CLASS_NAME}" YOLOV8_POSE_CLASS_NAME)

    if (NOT YOLOV8_POSE_CLASS_ID EQUAL YOLOV8_POSE_EXPECTED_CLASS_ID)
        message(FATAL_ERROR
            "YOLOv8 pose label IDs must be contiguous from zero; expected "
            "${YOLOV8_POSE_EXPECTED_CLASS_ID}, found ${YOLOV8_POSE_CLASS_ID}")
    endif()

    file(APPEND "${YOLOV8_POSE_LABELS_TXT_FILE}" "${YOLOV8_POSE_CLASS_NAME}\n")
    math(EXPR YOLOV8_POSE_EXPECTED_CLASS_ID
        "${YOLOV8_POSE_EXPECTED_CLASS_ID} + 1")
endforeach()

# Stage only supported images when the input is a directory. This prevents
# model artifacts stored beside samples from being misidentified by Pillow.
set(YOLOV8_POSE_IMAGE_SOURCE "${${use_case}_FILE_PATH}")
if (IS_DIRECTORY "${YOLOV8_POSE_IMAGE_SOURCE}")
    file(GLOB_RECURSE YOLOV8_POSE_IMAGE_FILES CONFIGURE_DEPENDS
        "${YOLOV8_POSE_IMAGE_SOURCE}/*.bmp"
        "${YOLOV8_POSE_IMAGE_SOURCE}/*.jpeg"
        "${YOLOV8_POSE_IMAGE_SOURCE}/*.jpg"
        "${YOLOV8_POSE_IMAGE_SOURCE}/*.png")
    if (NOT YOLOV8_POSE_IMAGE_FILES)
        message(FATAL_ERROR
            "No supported YOLOv8 pose images found in ${YOLOV8_POSE_IMAGE_SOURCE}")
    endif()

    set(YOLOV8_POSE_STAGED_IMAGE_DIR "${SAMPLES_GEN_DIR}/input")
    file(MAKE_DIRECTORY "${YOLOV8_POSE_STAGED_IMAGE_DIR}")
    file(COPY ${YOLOV8_POSE_IMAGE_FILES}
        DESTINATION "${YOLOV8_POSE_STAGED_IMAGE_DIR}")
    set(YOLOV8_POSE_IMAGE_SOURCE "${YOLOV8_POSE_STAGED_IMAGE_DIR}")
endif()

generate_images_code(
    "${YOLOV8_POSE_IMAGE_SOURCE}"
    ${SAMPLES_GEN_DIR}
    "${${use_case}_IMAGE_SIZE}")

set(${use_case}_LABELS_CPP_FILE Labels)
generate_labels_code(
    INPUT "${YOLOV8_POSE_LABELS_TXT_FILE}"
    DESTINATION_SRC ${SRC_GEN_DIR}
    DESTINATION_HDR ${INC_GEN_DIR}
    OUTPUT_FILENAME "${${use_case}_LABELS_CPP_FILE}")

if (ML_FRAMEWORK STREQUAL "${ML_FRAMEWORK_EXECUTORCH}" AND COMMAND generate_pte_ops_lib)
    generate_pte_ops_lib(
        MODEL_PATH "${${use_case}_MODEL_PATH}"
        LIB_NAME "${use_case}_portable_ops_lib"
        SELECT_OPS_LIST "")

    if (TARGET ${use_case}_portable_ops_lib)
        set(${use_case}_LINK_LIBS ${use_case}_portable_ops_lib)
    endif()
endif()

generate_model_code(
    MODEL_PATH ${${use_case}_MODEL_PATH}
    DESTINATION ${SRC_GEN_DIR}
    NAMESPACE "arm" "app" "yolov8_pose")

set(${use_case}_COMPILE_DEFS
    "YOLOV8_POSE_NUM_CLASSES=${${use_case}_NUM_CLASSES}"
    "YOLOV8_POSE_MAX_DETECTIONS=${${use_case}_MAX_DETECTIONS}"
    "YOLOV8_POSE_SCORE_THRESHOLD=${${use_case}_SCORE_THRESHOLD}F"
    "YOLOV8_POSE_NMS_THRESHOLD=${${use_case}_NMS_THRESHOLD}F"
    "YOLOV8_POSE_KEYPOINT_THRESHOLD=${${use_case}_KEYPOINT_THRESHOLD}F"
    "YOLOV8_POSE_COORDINATES_NORMALIZED=$<BOOL:${${use_case}_COORDINATES_NORMALIZED}>"
    "YOLOV8_POSE_EXPECTED_NUM_OUTPUTS=${YOLOV8_POSE_EXPECTED_NUM_OUTPUTS}"
    "YOLOV8_POSE_DISPLAY_DOWNSCALE=${${use_case}_DISPLAY_DOWNSCALE}"
    "YOLOV8_POSE_PMU_WARMUP_RUNS=${${use_case}_PMU_WARMUP_RUNS}"
    "YOLOV8_POSE_PMU_MEASURED_RUNS=${${use_case}_PMU_MEASURED_RUNS}")
