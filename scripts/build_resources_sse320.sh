#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
# Usage: bash scripts/build_resources_sse320.sh [all|tflm|pte] [all|dedicated|shared]
# Dedicated cache: 256 KiB. Shared: no separate cache. FPGA / U85 Z1024 / TA Mid.
# Requires a Linux Python environment, raw TFLite models and the GNU Arm toolchain.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
resource_root="${RESOURCE_ROOT:-$repo_root/resources_downloaded/linux-batch}"
venv_path="${MLEK_VENV:-$resource_root/env}"
build_root="${RESOURCE_BUILD_ROOT:-$repo_root/build/resources-sse320-release}"
package_root="${RESOURCE_PACKAGE_ROOT:-$repo_root/build/board-test-packages/resources-sse320-release}"
build_jobs="${BUILD_JOBS:-8}"
case "${1:-all}" in
    all) frameworks=(tflm pte) ;;
    tflm|pte) frameworks=("$1") ;;
    *) printf 'Framework must be all, tflm or pte.\n' >&2; exit 1 ;;
esac
case "${2:-all}" in
    all) variants=(dedicated shared) ;;
    dedicated|shared) variants=("$2") ;;
    *) printf 'Memory variant must be all, dedicated or shared.\n' >&2; exit 1 ;;
esac
[[ $# -le 2 && "$build_jobs" =~ ^[1-9][0-9]*$ ]] || exit 1
if [[ -n "${ARM_GNU_TOOLCHAIN:-}" ]]; then export PATH="$ARM_GNU_TOOLCHAIN/bin:$PATH"; fi
unset PYTHONHOME
export TORCH_HOME="$resource_root/torch-cache"
export PATH="$venv_path/bin:$PATH"
export PYTHONPATH="$repo_root/scripts/py${PYTHONPATH:+:$PYTHONPATH}"
python="$venv_path/bin/python"
cmake="$venv_path/bin/cmake"
for tool in arm-none-eabi-gcc arm-none-eabi-g++ arm-none-eabi-objcopy ninja; do
    command -v "$tool" >/dev/null || { printf 'Missing tool: %s\n' "$tool" >&2; exit 1; }
done
[[ -x "$python" && -x "$cmake" ]] || exit 1
use_cases=(ad asr img_class kws kws_asr noise_reduction object_detection vww)
vela_config="$repo_root/scripts/vela/default_vela.ini"
mkdir -p -- "$build_root" "$package_root"
common_options=(
    "-DCMAKE_TOOLCHAIN_FILE=$repo_root/scripts/cmake/toolchains/bare-metal-gcc.cmake"
    -DCMAKE_BUILD_TYPE=Release "-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG"
    "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG" "-DCMAKE_ASM_FLAGS_RELEASE=-O3 -DNDEBUG"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_SYSTEM_PROCESSOR=cortex-m85
    -DCMSIS_OPTIMIZATION_LEVEL=-O3
    -DTARGET_PLATFORM=mps4 -DTARGET_SUBSYSTEM=sse-320 -DFPGA_PLATFORM_SSE_320=ON
    "-DMLEK_PYTHON_VENV=$venv_path" -DETHOS_U_NPU_ENABLED=ON
    -DETHOS_U_NPU_ID=U85 -DETHOS_U_NPU_CONFIG_ID=Z1024
    -DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON
    "-DTA_CONFIG_FILE=$repo_root/scripts/cmake/timing_adapter/ta_config_u85_sys_dram_mid.cmake"
    -DUSE_SINGLE_INPUT=ON -DINTERACTIVE_MODE=OFF -DCPU_PROFILE_ENABLED=ON
)
failed=0
for framework in "${frameworks[@]}"; do
    for variant in "${variants[@]}"; do
        if [[ "$variant" == dedicated ]]; then
            memory_mode=Dedicated_Sram; cache_bytes=262144; arena_cache_bytes=262144
        else
            memory_mode=Shared_Sram; cache_bytes=0; arena_cache_bytes=4194304
        fi
        model_root="$resource_root/$variant"
        build_dir="$build_root/$framework-$variant"
        mkdir -p -- "$model_root" "$build_dir"
        model_options=(); extra_options=()
        if [[ "$framework" == tflm ]]; then
            selected=("${use_cases[@]}")
            for uc in "${selected[@]}"; do
                mkdir -p -- "$model_root/$uc"
                for raw_model in "$resource_root/raw/$uc/"*.tflite; do
                    [[ -f "$raw_model" ]] || { printf 'Missing raw model: %s\n' "$raw_model" >&2; exit 1; }
                    stem="$(basename -- "$raw_model" .tflite)"
                    output="$model_root/$uc/${stem}_vela_Z1024_${memory_mode}.tflite"
                    if [[ ! -f "$output" || ! -f "$output.sha256" ]] ||
                        ! (cd -- "$model_root/$uc" && sha256sum -c "$(basename -- "$output").sha256" >/dev/null); then
                        "$python" -m ethosu.vela "$raw_model" \
                            "--config=$vela_config" --accelerator-config=ethos-u85-1024 \
                            --system-config=Ethos_U85_SYS_DRAM_Mid_1024 \
                            "--memory-mode=$memory_mode" "--arena-cache-size=$arena_cache_bytes" \
                            --optimise=Performance --cop-format=COP1 \
                            "--output-dir=$model_root/$uc" 2>&1 | tee "$model_root/$uc/$stem-vela.log"
                        mv -- "$model_root/$uc/${stem}_vela.tflite" "$output"
                        (cd -- "$model_root/$uc" && sha256sum "$(basename -- "$output")" > "$(basename -- "$output").sha256")
                    fi
                done
            done
            ml_framework=tflm
            extra_options+=(-DTENSORFLOW_LITE_MICRO_BUILD_TYPE=release -DTFLM_OPTIMIZATION_LEVEL=-O3)
            if [[ "$variant" == shared ]]; then
                extra_options+=(-Dasr_ACTIVATION_BUF_SZ=0x00400000
                    -Dkws_asr_ACTIVATION_BUF_SZ=0x00400000)
            fi
        else
            selected=(img_class); ml_framework=executorch
            extra_options+=(-DEXECUTORCH_OPTIMIZE_SIZE=OFF
                "-DCMSIS_NN_LOCAL_PATH=$resource_root/host-cmsis-nn/CMSIS-NN-d933672e7ca97eec70ef43230baee7b20c2a28ae"
                "-DCMSIS_NN_SRC_PATH=$resource_root/host-cmsis-nn/CMSIS-NN-d933672e7ca97eec70ef43230baee7b20c2a28ae")
            model_dir="$model_root/img_class"
            mkdir -p -- "$model_dir"
            model="$model_dir/mv2_arm_delegate_ethos-u85-1024_${memory_mode}.pte"
            if [[ ! -f "$model" || ! -f "$model.sha256" ]] ||
                ! (cd -- "$model_dir" && sha256sum -c "$(basename -- "$model").sha256" >/dev/null); then
                (cd -- "$repo_root/dependencies/executorch" &&
                    "$python" "$repo_root/scripts/py/mlek_tools/lowering/executorch_arm_backend_lowering.py" \
                        --model_name=mv2 --target=ethos-u85-1024 \
                        --system_config=Ethos_U85_SYS_DRAM_Mid_1024 "--memory_mode=$memory_mode" \
                        "--config=$vela_config" "--extra-vela-flag=--arena-cache-size=$arena_cache_bytes" \
                        --extra-vela-flag=--cop-format=COP1 --extra-vela-flag=--optimise=Performance \
                        --delegate --quantize "--output=$model_dir") \
                    2>&1 | tee "$model_dir/pte-export.log"
                (cd -- "$model_dir" && sha256sum "$(basename -- "$model")" > "$(basename -- "$model").sha256")
            fi
            model_options+=("-Dimg_class_MODEL_PATH=$model")
            if [[ "$variant" == shared ]]; then
                # Method pool: SRAM [0x31000000, 0x31200000); scratch: following 2 MiB.
                extra_options+=(-DML_FWK_TMP_MEM_BASE=0x31200000 -DML_FWK_TMP_MEM_SIZE=0x00200000
                    -DHDLCD_FRAME_BUFFER_BASE_ADDRESS=0x74000000)
            fi
        fi
        use_case_list="$(IFS=';'; printf '%s' "${selected[*]}")"
        configure=("$cmake" -S "$repo_root" -B "$build_dir" -G Ninja
            "${common_options[@]}" "${extra_options[@]}" "${model_options[@]}"
            "-DRESOURCES_PATH=$model_root" "-DML_FRAMEWORK=$ml_framework"
            "-DUSE_CASE_BUILD=$use_case_list" "-DETHOS_U_NPU_MEMORY_MODE=$memory_mode"
            "-DETHOS_U_NPU_CACHE_SIZE=$cache_bytes")
        printf '\nConfiguring %s / %s\n' "$framework" "$variant"
        printf '%q ' "${configure[@]}"; printf '\n'
        if ! "${configure[@]}" 2>&1 | tee "$build_dir/configure.log"; then
            failed=1; continue
        fi
        for uc in "${selected[@]}"; do
            printf '\nBuilding %s / %s / %s\n' "$uc" "$framework" "$variant"
            if ! "$cmake" --build "$build_dir" --target "mlek_$uc" --parallel "$build_jobs" \
                2>&1 | tee "$build_dir/$uc-build.log"; then
                failed=1; continue
            fi
            destination="$package_root/$uc/$framework-$variant"
            mkdir -p -- "$destination/SOFTWARE"
            cp -- "$build_dir/bin/sectors/$uc/"{boot,bram,ddr}.bin "$destination/SOFTWARE/"
            cp -- "$build_dir/bin/sectors/images.txt" "$destination/"
            cp -- "$build_dir/bin/mlek_$uc.axf" "$build_dir/bin/mlek_$uc.map" "$destination/"
            {
                printf 'use_case=%s\nframework=%s\nmemory_mode=%s\nnpu_cache_bytes=%s\n' \
                    "$uc" "$framework" "$memory_mode" "$cache_bytes"
                printf 'build_type=Release\noptimization=-O3\nplatform=MPS4 SSE-320 FPGA\n'
                printf 'npu=U85 Z1024\nsystem_config=Ethos_U85_SYS_DRAM_Mid_1024\n'
                printf 'single_input=ON\ninteractive=OFF\nbuild_directory=%s\n' "$build_dir"
                printf 'git_commit=%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
                "$python" -c 'import importlib.metadata as m; print("vela_version=" + m.version("ethos-u-vela"))'
                if [[ "$framework" == pte ]]; then
                    cat "$model.sha256"
                else
                    cat "$model_root/$uc/"*.tflite.sha256
                fi
                printf 'configure_command='; printf '%q ' "${configure[@]}"; printf '\n'
            } > "$destination/build-info.txt"
            (cd -- "$destination" && sha256sum SOFTWARE/*.bin "mlek_$uc.axf" > SHA256SUMS)
            printf 'Package: %s\n' "$destination"
        done
    done
done
exit "$failed"
