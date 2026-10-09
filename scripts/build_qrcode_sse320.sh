#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

# Usage: bash scripts/build_qrcode_sse320.sh [all|fvp|fpga] [all|variant]
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
deployment_selection="${1:-all}"
variant_selection="${2:-all}"
build_jobs="${BUILD_JOBS:-8}"
venv_path="$repo_root/resources_downloaded/env"
model_root="${QRCODE_MODEL_ROOT:-$repo_root/resources_downloaded/yolo_QRcode}"
fastcache_kib="${QRCODE_FASTCACHE_KIB:-}"
fastcache_root="${QRCODE_FASTCACHE_ROOT:-$model_root/fastcache_dedicated}"
sample_path="${QRCODE_SAMPLE_PATH:-$repo_root/resources/object_detection/samples_QRcode/1.png}"
labels_file="${QRCODE_LABELS_FILE:-$model_root/labels.txt}"
cmake_exe="$venv_path/bin/cmake"

if (( $# > 2 )); then
    printf 'Usage: bash %s [all|fvp|fpga] [all|tflm-shared|tflm-dedicated|pte-shared|pte-dedicated]\n' \
        "${BASH_SOURCE[0]}" >&2
    exit 1
fi
case "$deployment_selection" in
    all) deployments=(fvp fpga) ;;
    fvp|fpga) deployments=("$deployment_selection") ;;
    *) printf 'Unsupported deployment: %s\n' "$deployment_selection" >&2; exit 1 ;;
esac
case "$variant_selection" in
    all) variants=(tflm-shared tflm-dedicated pte-dedicated pte-shared) ;;
    tflm-shared|tflm-dedicated|pte-shared|pte-dedicated) variants=("$variant_selection") ;;
    *) printf 'Unsupported variant: %s\n' "$variant_selection" >&2; exit 1 ;;
esac
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'BUILD_JOBS must be a positive integer.\n' >&2
    exit 1
fi
case "$fastcache_kib" in
    ""|64|128|256|384|768) ;;
    *) printf 'QRCODE_FASTCACHE_KIB must be 64, 128, 256, 384 or 768.\n' >&2; exit 1 ;;
esac
if [[ ! -x "$cmake_exe" || ! -x "$venv_path/bin/python" || ! -f "$labels_file" ||
      ! -e "$sample_path" ]]; then
    printf 'Project environment, labels or input sample is missing.\n' >&2
    exit 1
fi

model_dimensions="$("$venv_path/bin/python" - "$model_root" <<'PY'
import json
from pathlib import Path
import sys
report = json.loads((Path(sys.argv[1]) / "export_report.json").read_text())
if report["status"] != "completed":
    raise SystemExit("Models must pass export validation before building firmware")
config = report["configuration"]
if (config["target"] != "ethos-u85-1024"
        or config["system_config"] != "Ethos_U85_SYS_DRAM_Mid_1024"
        or config["arena_cache_size"] != 393216
        or config["preprocessing"] != "mlek-crop"):
    raise SystemExit("Models must use U85 Z1024 / TA Mid / cache 393216 / mlek-crop")
print(report["contract"]["image_size"], report["contract"]["num_classes"])
PY
)"
read -r image_size num_classes <<< "$model_dimensions"
unset PYTHONHOME
export PATH="$venv_path/bin:$PATH"
for tool in arm-none-eabi-gcc arm-none-eabi-g++ arm-none-eabi-objcopy make; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Required tool is missing: %s\n' "$tool" >&2
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
    "-Dyolov8_detection_IMAGE_SIZE=$image_size"
    "-Dyolov8_detection_NUM_CLASSES=$num_classes"
    "-Dyolov8_detection_FILE_PATH=$sample_path"
    "-Dyolov8_detection_LABELS_YAML_FILE=$labels_file"
)

for deployment in "${deployments[@]}"; do
    fpga_option=OFF
    if [[ "$deployment" == fpga ]]; then
        fpga_option=ON
    fi
    for variant in "${variants[@]}"; do
        memory_options=()
        case "$variant" in
            tflm-shared|tflm-dedicated)
                framework=tflm
                arena_size=0x00400000
                if [[ "$variant" == tflm-shared ]]; then
                    memory_mode=Shared_Sram
                else
                    memory_mode=Dedicated_Sram
                    memory_options=(-DETHOS_U_NPU_CACHE_SIZE=393216)
                fi
                model="$model_root/tflm_${memory_mode}/best_int8_vela.tflite"
                ;;
            pte-dedicated|pte-shared)
                framework=executorch
                arena_size=0x00300000
                if [[ "$variant" == pte-dedicated ]]; then
                    memory_mode=Dedicated_Sram
                    memory_options=(-DETHOS_U_NPU_CACHE_SIZE=393216)
                else
                    memory_mode=Shared_Sram
                    memory_options=(
                        -DETHOS_U_NPU_CACHE_SIZE=0
                        -Dyolov8_detection_ET_METHOD_MEMORY_IN_DDR=ON
                        -DML_FWK_TMP_MEM_BASE=0x31000000
                        -DML_FWK_TMP_MEM_SIZE=0x00400000
                    )
                    if [[ "$deployment" == fpga ]]; then
                        memory_options+=(-DHDLCD_FRAME_BUFFER_BASE_ADDRESS=0x74000000)
                    else
                        memory_options+=(-DHDLCD_FRAME_BUFFER_BASE_ADDRESS=0x96000000)
                    fi
                fi
                model="$model_root/pte_${memory_mode}/best.pte"
                ;;
        esac
        if [[ -n "$fastcache_kib" && "$memory_mode" == Dedicated_Sram ]]; then
            cache_bytes=$((fastcache_kib * 1024))
            model_framework=tflm
            model_name=best_int8_vela.tflite
            if [[ "$framework" == executorch ]]; then
                model_framework=pte
                model_name=best.pte
            fi
            directory="$fastcache_root/$model_framework/${fastcache_kib}k"
            model="$directory/$model_name"
            "$venv_path/bin/python" - "$directory" "$cache_bytes" "$repo_root" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
record = json.loads((Path(sys.argv[1]) / "compilation.json").read_text())
if (record["status"] != "completed" or record["cache_bytes"] != int(sys.argv[2])
        or record["cpu_operators"] or record["npu_delegate_count"] != 1
        or record["target"] != "ethos-u85-1024"
        or record["system_config"] != "Ethos_U85_SYS_DRAM_Mid_1024"
        or record["memory_mode"] != "Dedicated_Sram"):
    raise SystemExit("Fast-cache model configuration or NPU-only check failed")
config = Path(sys.argv[3]) / "scripts/vela/default_vela.ini"
if hashlib.sha256(config.read_bytes()).hexdigest() != record["vela_config_sha256"]:
    raise SystemExit("Vela configuration changed; recompile the cache model")
if hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() != record["model_sha256"]:
    raise SystemExit("Fast-cache model checksum mismatch; recompile")
PY
            memory_options=("-DETHOS_U_NPU_CACHE_SIZE=$cache_bytes")
            build_dir="$repo_root/build-qrcode-fastcache-${deployment}-${model_framework}-${fastcache_kib}k-z1024-ta-mid"
        else
            build_dir="$repo_root/build-qrcode-${deployment}-${variant}-z1024-ta-mid"
        fi
        if [[ ! -f "$model" ]]; then
            printf 'Compiled model is missing: %s\n' "$model" >&2
            exit 1
        fi
        mkdir -p -- "$build_dir"
        printf '\nBuilding QR-code %s / %s\n' "$deployment" "$variant"
        "$cmake_exe" --fresh -S "$repo_root" -B "$build_dir" -G "Unix Makefiles" \
            "${common_options[@]}" "${memory_options[@]}" \
            "-DFPGA_PLATFORM_SSE_320=$fpga_option" "-DML_FRAMEWORK=$framework" \
            "-DETHOS_U_NPU_MEMORY_MODE=$memory_mode" \
            "-Dyolov8_detection_MODEL_PATH=$model" \
            "-Dyolov8_detection_ACTIVATION_BUF_SZ=$arena_size" 2>&1 | tee "$build_dir/configure.log"
        "$cmake_exe" --build "$build_dir" \
            --target mlek_yolov8_detection --parallel "$build_jobs" 2>&1 | tee "$build_dir/build.log"
        printf 'Firmware: %s/bin/mlek_yolov8_detection.axf\n' "$build_dir"
        if [[ "$deployment" == fpga ]]; then
            printf 'FPGA sectors: %s/bin/sectors/\n' "$build_dir"
        fi
    done
done
