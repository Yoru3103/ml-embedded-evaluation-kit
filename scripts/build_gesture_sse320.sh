#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

# Usage: bash scripts/build_gesture_sse320.sh [all|fvp|fpga] [variant|all]
# Variants: tflm-shared, tflm-dedicated, pte-dedicated, pte-shared
# "all" retains the original three variants; request pte-shared explicitly.
# BUILD_JOBS defaults to 8. Each configuration has its own build directory.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
target_selection="${1:-all}"
variant_selection="${2:-all}"
build_jobs="${BUILD_JOBS:-8}"

if (( $# > 2 )); then
    printf 'Usage: bash %s [all|fvp|fpga] [all|tflm-shared|tflm-dedicated|pte-dedicated|pte-shared]\n' \
        "${BASH_SOURCE[0]}" >&2
    exit 1
fi

case "$target_selection" in
    all) deployments=(fvp fpga) ;;
    fvp|fpga) deployments=("$target_selection") ;;
    *) printf 'Unsupported deployment: %s\n' "$target_selection" >&2; exit 1 ;;
esac

case "$variant_selection" in
    all) variants=(tflm-shared tflm-dedicated pte-dedicated) ;;
    tflm-shared|tflm-dedicated|pte-dedicated|pte-shared) variants=("$variant_selection") ;;
    *) printf 'Unsupported variant: %s\n' "$variant_selection" >&2; exit 1 ;;
esac

if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'BUILD_JOBS must be a positive integer.\n' >&2
    exit 1
fi

venv_path="$repo_root/resources_downloaded/env"
cmake_exe="$venv_path/bin/cmake"
model_root="$repo_root/resources_downloaded/gesture_detection"

if [[ ! -x "$cmake_exe" || ! -x "$venv_path/bin/python3" ]]; then
    printf 'Project Python/CMake environment is missing: %s\n' "$venv_path" >&2
    exit 1
fi

# Keep the project's Python 3.12 environment separate from FVP's embedded Python.
unset PYTHONHOME
export PATH="$venv_path/bin:$PATH"

for tool in arm-none-eabi-gcc arm-none-eabi-g++ arm-none-eabi-objcopy make; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Required tool is missing from PATH: %s\n' "$tool" >&2
        exit 1
    fi
done

common_options=(
    "-DCMAKE_TOOLCHAIN_FILE=$repo_root/scripts/cmake/toolchains/bare-metal-gcc.cmake"
    -DCMAKE_BUILD_TYPE=Release
    -DTARGET_PLATFORM=mps4
    -DTARGET_SUBSYSTEM=sse-320
    "-DMLEK_PYTHON_VENV=$venv_path"
    "-DRESOURCES_PATH=$repo_root/resources_downloaded"
    -DETHOS_U_NPU_ENABLED=ON
    -DETHOS_U_NPU_ID=U85
    -DETHOS_U_NPU_CONFIG_ID=Z1024
    -DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON
    "-DTA_CONFIG_FILE=$repo_root/scripts/cmake/timing_adapter/ta_config_u85_sys_dram_mid.cmake"
    -DUSE_CASE_BUILD=yolov8_detection
    -DUSE_SINGLE_INPUT=ON
    -DINTERACTIVE_MODE=OFF
    -DCPU_PROFILE_ENABLED=ON
    -Dyolov8_detection_IMAGE_SIZE=320
    -Dyolov8_detection_NUM_CLASSES=10
    "-Dyolov8_detection_FILE_PATH=$repo_root/resources/gesture_detection/samples"
    "-Dyolov8_detection_LABELS_YAML_FILE=$repo_root/resources/gesture_detection/labels.txt"
)

for deployment in "${deployments[@]}"; do
    fpga_option=OFF
    if [[ "$deployment" == fpga ]]; then
        fpga_option=ON
    fi

    for variant in "${variants[@]}"; do
        memory_options=()
        case "$variant" in
            tflm-shared)
                framework=tflm
                memory_mode=Shared_Sram
                arena_size=0x00400000
                model="$model_root/tflm_Shared_Sram/best_int8_vela.tflite"
                ;;
            tflm-dedicated)
                framework=tflm
                memory_mode=Dedicated_Sram
                arena_size=0x00400000
                model="$model_root/tflm_Dedicated_Sram/best_int8_vela.tflite"
                memory_options=(-DETHOS_U_NPU_CACHE_SIZE=393216)
                ;;
            pte-dedicated)
                framework=executorch
                memory_mode=Dedicated_Sram
                arena_size=0x00300000
                model="$model_root/pte_Dedicated_Sram/best.pte"
                memory_options=(-DETHOS_U_NPU_CACHE_SIZE=393216)
                ;;
            pte-shared)
                framework=executorch
                memory_mode=Shared_Sram
                arena_size=0x00300000
                model="$model_root/pte_Shared_Sram/best.pte"
                # SSE-320 has 4 MiB of SRAM. Reserve it for NPU scratch;
                # the separate method pool also holds the float CPU input.
                memory_options=(
                    -DETHOS_U_NPU_CACHE_SIZE=0
                    -Dyolov8_detection_ET_METHOD_MEMORY_IN_DDR=ON
                    -DML_FWK_TMP_MEM_BASE=0x31000000
                    -DML_FWK_TMP_MEM_SIZE=0x00400000
                )
                # The platform normally derives the display address from
                # the temporary pool. Keep it in DDR when that pool is SRAM.
                if [[ "$deployment" == fpga ]]; then
                    memory_options+=(-DHDLCD_FRAME_BUFFER_BASE_ADDRESS=0x74000000)
                else
                    memory_options+=(-DHDLCD_FRAME_BUFFER_BASE_ADDRESS=0x96000000)
                fi
                ;;
        esac

        if [[ ! -f "$model" ]]; then
            printf 'Compiled model is missing: %s\n' "$model" >&2
            exit 1
        fi

        build_dir="$repo_root/build-yolo-${deployment}-${variant}-z1024-ta-mid"
        printf '\nBuilding %s / %s with TA Mid\n' "$deployment" "$variant"

        # Refresh CMake configuration so cached TA register values cannot persist.
        "$cmake_exe" --fresh -S "$repo_root" -B "$build_dir" -G "Unix Makefiles" \
            "${common_options[@]}" \
            "${memory_options[@]}" \
            "-DFPGA_PLATFORM_SSE_320=$fpga_option" \
            "-DML_FRAMEWORK=$framework" \
            "-DETHOS_U_NPU_MEMORY_MODE=$memory_mode" \
            "-Dyolov8_detection_MODEL_PATH=$model" \
            "-Dyolov8_detection_ACTIVATION_BUF_SZ=$arena_size"

        "$cmake_exe" --build "$build_dir" \
            --target mlek_yolov8_detection --parallel "$build_jobs"

        printf 'Firmware: %s/bin/mlek_yolov8_detection.axf\n' "$build_dir"
        if [[ "$deployment" == fpga ]]; then
            printf 'FPGA images: %s/bin/sectors/\n' "$build_dir"
        fi
    done
done
