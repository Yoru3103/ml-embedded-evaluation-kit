#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

# Build isolated FPGA/FVP firmware using the Dedicated_Sram fast-cache models.
set -euo pipefail

usage()
{
    cat <<'HELP'
Usage: bash scripts/build_gesture_fastcache_sse320.sh [all|fpga|fvp] [all|tflm|pte] [all|64|128|384|768] [--dry-run]

Defaults: FPGA and FVP, both frameworks, all cache sizes (KiB): 16 firmware builds.
Run compile_gesture_fastcache.sh first. Models are shared between FPGA and FVP.
Examples:
  bash scripts/build_gesture_fastcache_sse320.sh
  bash scripts/build_gesture_fastcache_sse320.sh fpga pte 128
  bash scripts/build_gesture_fastcache_sse320.sh all all all --dry-run

Fixed settings: Dedicated_Sram, U85 Z1024, TA Mid, 320px/10 classes, same samples.
FPGA sets FPGA_PLATFORM_SSE_320=ON, FVP sets it OFF (including register selection).
TFLM arena: 4 MiB; PTE method pool: 3 MiB. Both reside in DDR in Dedicated_Sram.
NPU scratch: 32 MiB in each platform's default DDR region. Fast cache is in SRAM;
only its capacity is varied. 768 KiB is twice the baseline, within the 4 MiB SRAM.
Model configuration and SHA256 checks must pass before any build starts.

Outputs: build-gesture-fastcache-<fpga|fvp>-<tflm|pte>-<size>k-z1024-ta-mid/
  bin/mlek_yolov8_detection.axf
  bin/sectors/yolov8_detection/{boot,bram,ddr}.bin (FPGA only)
  configure.log, build.log, build-info.txt
Firmware selection: each directory embeds its corresponding model and cache size.
This script builds firmware; board loading and FVP performance runs are separate.

Environment overrides:
  BUILD_JOBS          Parallel compiler jobs (default 8)
  FASTCACHE_ROOT      Same absolute model output directory as compile script
  FASTCACHE_BUILD_ROOT  Absolute parent for build directories (default repository)
  MLEK_VENV           Absolute project Python/CMake environment path
  ARM_GNU_TOOLCHAIN   Absolute toolchain root containing bin/ (otherwise use PATH)
HELP
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dry_run=0
positionals=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=1 ;;
        -h|--help) usage; exit 0 ;;
        --*) printf 'Unknown option: %s\n' "$arg" >&2; exit 1 ;;
        *) positionals+=("$arg") ;;
    esac
done
if (( ${#positionals[@]} > 3 )); then usage >&2; exit 1; fi
case "${positionals[0]:-all}" in
    all) deployments=(fpga fvp) ;;
    fpga|fvp) deployments=("${positionals[0]}") ;;
    *) printf 'Deployment must be all, fpga or fvp.\n' >&2; exit 1 ;;
esac
case "${positionals[1]:-all}" in
    all) frameworks=(tflm pte) ;;
    tflm|pte) frameworks=("${positionals[1]}") ;;
    *) printf 'Framework must be all, tflm or pte.\n' >&2; exit 1 ;;
esac
case "${positionals[2]:-all}" in
    all) cache_sizes=(64 128 384 768) ;;
    64|128|384|768) cache_sizes=("${positionals[2]}") ;;
    *) printf 'Cache size must be all, 64, 128, 384 or 768 (KiB).\n' >&2; exit 1 ;;
esac
build_jobs="${BUILD_JOBS:-12}"
[[ "$build_jobs" =~ ^[1-9][0-9]*$ ]] || {
    printf 'BUILD_JOBS must be a positive integer.\n' >&2; exit 1;
}
model_root="${FASTCACHE_ROOT:-$repo_root/resources_downloaded/gesture_detection/fastcache_dedicated}"
build_root="${FASTCACHE_BUILD_ROOT:-$repo_root}"
venv_path="${MLEK_VENV:-$repo_root/resources_downloaded/env}"
cmake_exe="$venv_path/bin/cmake"
for path in "$model_root" "$build_root" "$venv_path"; do
    [[ "$path" == /* ]] || { printf 'Environment paths must be absolute: %s\n' "$path" >&2; exit 1; }
done
unset PYTHONHOME
export PATH="$venv_path/bin:$PATH"
if [[ -n "${ARM_GNU_TOOLCHAIN:-}" ]]; then
    [[ "$ARM_GNU_TOOLCHAIN" == /* ]] || { printf 'ARM_GNU_TOOLCHAIN must be absolute.\n' >&2; exit 1; }
    export PATH="$ARM_GNU_TOOLCHAIN/bin:$PATH"
fi
vela_config="$repo_root/scripts/vela/default_vela.ini"

model_path()
{
    local directory="$model_root/$1/${2}k"
    local checksum_file="$directory/model.sha256"
    # sha256sum uses two spaces before the filename; retain spaces in custom source names.
    local entry
    IFS= read -r entry < "$checksum_file"
    printf '%s/%s\n' "$directory" "${entry:66}"
}

if (( ! dry_run )); then
    [[ -x "$cmake_exe" && -x "$venv_path/bin/python3" ]] || {
        printf 'Python/CMake environment missing: %s\n' "$venv_path" >&2; exit 1;
    }
    for tool in arm-none-eabi-gcc arm-none-eabi-g++ arm-none-eabi-objcopy make; do
        command -v "$tool" >/dev/null 2>&1 || {
            printf 'Required tool missing from PATH: %s (or set ARM_GNU_TOOLCHAIN)\n' "$tool" >&2
            exit 1
        }
    done
    vela_hash="$(sha256sum "$vela_config" | cut -d ' ' -f 1)"
    # Preflight every selected model, so missing/mismatched cases fail before costly builds.
    for framework in "${frameworks[@]}"; do
        for cache_kib in "${cache_sizes[@]}"; do
            directory="$model_root/$framework/${cache_kib}k"
            cache_bytes=$((cache_kib * 1024))
            if [[ ! -f "$directory/configuration.txt" || ! -f "$directory/model.sha256" ]]; then
                printf 'Compile %s / %s KiB first: %s\n' "$framework" "$cache_kib" "$directory" >&2
                exit 1
            fi
            if ! cmp -s "$directory/configuration.txt" <(
                printf 'framework=%s\ncache_bytes=%s\n' "$framework" "$cache_bytes"
                printf 'memory_mode=Dedicated_Sram\ntarget=ethos-u85-1024\n'
                printf 'system_config=Ethos_U85_SYS_DRAM_Mid_1024\n'
                printf 'vela_config_sha256=%s\n' "$vela_hash"
            ); then
                printf 'Model configuration mismatch; recompile: %s\n' "$directory" >&2; exit 1
            fi
            (cd -- "$directory" && sha256sum --check --strict model.sha256)
            model="$(model_path "$framework" "$cache_kib")"
            case "$framework:$model" in
                tflm:*.tflite|pte:*.pte) ;;
                *) printf 'Unexpected model filename: %s\n' "$model" >&2; exit 1 ;;
            esac
        done
    done
fi

# Dedicated_Sram already places the activation/method pool in DDR. Keep the
# explicit method-pool override off to match the existing dedicated builds.
common_options=(
    "-DCMAKE_TOOLCHAIN_FILE=$repo_root/scripts/cmake/toolchains/bare-metal-gcc.cmake"
    -DCMAKE_BUILD_TYPE=Release -DTARGET_PLATFORM=mps4 -DTARGET_SUBSYSTEM=sse-320
    "-DMLEK_PYTHON_VENV=$venv_path" "-DRESOURCES_PATH=$repo_root/resources_downloaded"
    -DETHOS_U_NPU_ENABLED=ON -DETHOS_U_NPU_ID=U85 -DETHOS_U_NPU_CONFIG_ID=Z1024
    -DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram -DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON
    "-DTA_CONFIG_FILE=$repo_root/scripts/cmake/timing_adapter/ta_config_u85_sys_dram_mid.cmake"
    -DUSE_CASE_BUILD=yolov8_detection -DUSE_SINGLE_INPUT=ON -DINTERACTIVE_MODE=OFF
    -DCPU_PROFILE_ENABLED=ON -Dyolov8_detection_IMAGE_SIZE=320
    -Dyolov8_detection_NUM_CLASSES=10 -Dyolov8_detection_ET_METHOD_MEMORY_IN_DDR=OFF
    "-Dyolov8_detection_FILE_PATH=$repo_root/resources/gesture_detection/samples"
    "-Dyolov8_detection_LABELS_YAML_FILE=$repo_root/resources/gesture_detection/labels.txt"
)

for deployment in "${deployments[@]}"; do
    fpga_option=OFF
    if [[ "$deployment" == fpga ]]; then fpga_option=ON; fi
    for framework in "${frameworks[@]}"; do
        ml_framework=tflm
        arena_size=0x00400000
        if [[ "$framework" == pte ]]; then
            ml_framework=executorch
            arena_size=0x00300000
        fi
        for cache_kib in "${cache_sizes[@]}"; do
            cache_bytes=$((cache_kib * 1024))
            directory="$model_root/$framework/${cache_kib}k"
            if (( dry_run )); then
                model="$directory/best_int8_vela.tflite"
                if [[ "$framework" == pte ]]; then model="$directory/best.pte"; fi
            else
                model="$(model_path "$framework" "$cache_kib")"
            fi
            build_dir="$build_root/build-gesture-fastcache-${deployment}-${framework}-${cache_kib}k-z1024-ta-mid"
            configure=("$cmake_exe" --fresh -S "$repo_root" -B "$build_dir" -G "Unix Makefiles"
                "${common_options[@]}" "-DFPGA_PLATFORM_SSE_320=$fpga_option"
                "-DML_FRAMEWORK=$ml_framework" "-DETHOS_U_NPU_CACHE_SIZE=$cache_bytes"
                "-Dyolov8_detection_MODEL_PATH=$model"
                "-Dyolov8_detection_ACTIVATION_BUF_SZ=$arena_size")
            build=("$cmake_exe" --build "$build_dir" --target mlek_yolov8_detection
                --parallel "$build_jobs")
            printf '\nBuild %s / %s / %s KiB (%s bytes)\n' \
                "$deployment" "$framework" "$cache_kib" "$cache_bytes"
            printf '%q ' "${configure[@]}"; printf '\n'
            printf '%q ' "${build[@]}"; printf '\n'
            if (( dry_run )); then continue; fi
            mkdir -p -- "$build_dir"
            {
                printf 'deployment=%s\n' "$deployment"
                cat "$directory/configuration.txt" "$directory/model.sha256"
                printf 'model=%s\n' "$model"
                printf 'git_commit=%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
                git -C "$repo_root" status --short
                printf '%q ' "${configure[@]}"; printf '\n'
                printf '%q ' "${build[@]}"; printf '\n'
            } > "$build_dir/build-info.txt"
            "${configure[@]}" 2>&1 | tee "$build_dir/configure.log"
            "${build[@]}" 2>&1 | tee "$build_dir/build.log"
            printf 'Firmware: %s/bin/mlek_yolov8_detection.axf\n' "$build_dir"
            if [[ "$deployment" == fpga ]]; then
                printf 'FPGA images: %s/bin/sectors/yolov8_detection/\n' "$build_dir"
            fi
        done
    done
done
