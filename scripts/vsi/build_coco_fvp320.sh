#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

# Build the alif 256-pixel COCO detector with the MLEK KEY0/VSI camera flow.
set -euo pipefail

# FVP runtime.sh exports its bundled Python 3.9 home; builds use the project venv.
unset PYTHONHOME

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${COCO_BUILD_DIR:-$repo_root/build-coco-vsi}"
python_path="${MLEK_PYTHON:-$repo_root/resources_downloaded/env/bin/python}"
model_dir="$repo_root/resources_downloaded/coco_detection"
source_model="$model_dir/source/best_int8.tflite"
compiled_dir="$model_dir/tflm_Dedicated_Sram"
compiled_model="$compiled_dir/best_int8_vela.tflite"
vela_config="$repo_root/scripts/vela/default_vela.ini"

if [[ ! -x "$python_path" ]]; then
    printf 'Project Python environment is missing: %s\n' "$python_path" >&2
    exit 1
fi

# Import once; subsequent builds use only the local MLEK model and resources.
if [[ ! -f "$source_model" ]]; then
    alif_root="${ALIF_ROOT:-$HOME/alif_ml-embedded-evaluation-kit}"
    import_model="${COCO_MODEL_SOURCE:-$alif_root/20260827_Model/best_int8.tflite}"
    if [[ ! -f "$import_model" ]]; then
        printf 'Set COCO_MODEL_SOURCE to the alif 256x256, 80-class best_int8.tflite.\n' >&2
        exit 1
    fi
    mkdir -p "$(dirname -- "$source_model")"
    cp -- "$import_model" "$source_model"
fi

mkdir -p "$build_dir" "$compiled_dir"
"$python_path" -m ethosu.vela "$source_model" \
    --accelerator-config=ethos-u85-1024 \
    --config="$vela_config" \
    --system-config=Ethos_U85_SYS_DRAM_Mid_1024 \
    --memory-mode=Dedicated_Sram \
    --arena-cache-size=393216 \
    --show-cpu-operations \
    --output-dir="$compiled_dir" 2>&1 | tee "$build_dir/vela.log"

sha256sum "$source_model" "$compiled_model" "$vela_config" \
    "$repo_root/resources/coco_detection/coco128.yaml" > "$build_dir/model_sha256.txt"

export PATH="$(dirname -- "$python_path"):$PATH"
cmake -S "$repo_root" -B "$build_dir" \
    -DCMAKE_TOOLCHAIN_FILE="$repo_root/scripts/cmake/toolchains/bare-metal-gcc.cmake" \
    -DTARGET_PLATFORM=mps4 \
    -DTARGET_SUBSYSTEM=sse-320 \
    -DFPGA_PLATFORM_SSE_320=OFF \
    -DML_FRAMEWORK=tflm \
    -DUSE_CASE_BUILD=yolov8_detection \
    -DUSE_SINGLE_INPUT=ON \
    -DETHOS_U_NPU_ENABLED=ON \
    -DETHOS_U_NPU_ID=U85 \
    -DETHOS_U_NPU_CONFIG_ID=Z1024 \
    -DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram \
    -DETHOS_U_NPU_CACHE_SIZE=393216 \
    -DFVP_VSI_ENABLED=ON \
    -DBUILD_FVP_TESTS=OFF \
    -DINTERACTIVE_MODE=OFF \
    -DSEMIHOSTING_ENABLED=OFF \
    -Dyolov8_detection_CAMERA_INPUT=ON \
    -Dyolov8_detection_BUTTON_CAPTURE=ON \
    -Dyolov8_detection_MODEL_PATH="$compiled_model" \
    -Dyolov8_detection_LABELS_YAML_FILE="$repo_root/resources/coco_detection/coco128.yaml" \
    -Dyolov8_detection_FILE_PATH="$repo_root/resources/coco_detection/samples/000000058350.jpg" \
    -Dyolov8_detection_IMAGE_SIZE=256 \
    -Dyolov8_detection_NUM_CLASSES=80 \
    -Dyolov8_detection_SCORE_THRESHOLD=0.1 \
    -Dyolov8_detection_NMS_THRESHOLD=0.45 \
    -Dyolov8_detection_MAX_DETECTIONS=20 \
    -Dyolov8_detection_DISPLAY_DOWNSCALE=2 \
    -Dyolov8_detection_ACTIVATION_BUF_SZ=0x00200000 \
    -Dyolov8_detection_BOX_COORDINATES=normalized \
    -Dyolov8_detection_INPUT_IS_GRAYSCALE=OFF \
    -Dyolov8_detection_OBB=OFF

cmake --build "$build_dir" --target mlek_yolov8_detection -j"${BUILD_JOBS:-8}"
