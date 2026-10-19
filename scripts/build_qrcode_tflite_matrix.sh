#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

# Build 2 platforms x (Shared SRAM + 5 Dedicated SRAM fast-cache sizes).
# Supply an uncompiled, fully integer TFLite model with the qrcode tensor contract.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="$repo_root/resources_downloaded/env"
python="$venv_path/bin/python"
cmake="$venv_path/bin/cmake"
sample_path="${QRCODE_SAMPLE_PATH:-$repo_root/resources/qrcode/smaples}"
labels="${QRCODE_LABELS_FILE:-$repo_root/resources/qrcode/labels.txt}"
output="${QRCODE_MATRIX_ROOT:-$repo_root/build-qrcode-tflite-matrix}"
jobs="${BUILD_JOBS:-8}"
fvp_timeout="${FVP_TIMEOUT:-1800}"
fvp="${FVP_PATH:-$HOME/FVP_Corstone_SSE-320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320}"
dry_run=0
run_fvp=0
models_only=0
npu_model_root=""
model=""
output_type=int8
usage() {
    cat <<'HELP'
Usage: bash scripts/build_qrcode_tflite_matrix.sh MODEL.tflite [--run-fvp] [--dry-run]
   or: bash scripts/build_qrcode_tflite_matrix.sh --npu-model-root DIR [--run-fvp]

MODEL must be an INT8 TFLite BEFORE Vela compilation (not *_vela.tflite).
Builds FPGA and FVP firmware for Shared_Sram and Dedicated_Sram 64/128/256/384/768 KiB.
Every firmware runs all images, without interaction. --run-fvp also executes the 6 FVPs.
FPGA outputs must be loaded onto the board separately.
--models-only compiles/checks all 6 NPU models, without firmware builds.
--output-type {int8,float32} selects output tensors (default: int8).
float32 appends four terminal DEQUANTIZE operators after Vela compilation;
the source model and --npu-model-root must still provide INT8 NPU models.
--npu-model-root reuses DIR/{shared,64,128,256,384,768}/*_vela.tflite
with their adjacent Vela summary CSVs. All 6 models are checked before building.

Models are written under OUTPUT/models/VARIANT/.
Build and runtime logs: OUTPUT/{fpga,fvp}/VARIANT/{configure,build,fvp}.log
(fvp.log is generated only for --run-fvp.)

Environment: QRCODE_SAMPLE_PATH, QRCODE_LABELS_FILE, QRCODE_MATRIX_ROOT,
             BUILD_JOBS (8), FVP_PATH, FVP_TIMEOUT (1800 seconds per firmware).
--dry-run prints commands without compiling, building, or executing FVP.
HELP
}
while (( $# )); do
    argument="$1"
    shift
    case "$argument" in
        --help|-h) usage; exit 0 ;;
        --dry-run) dry_run=1 ;;
        --run-fvp) run_fvp=1 ;;
        --models-only) models_only=1 ;;
        --output-type)
            if (( $# == 0 )); then usage >&2; exit 1; fi
            output_type="$1"
            shift
            ;;
        --npu-model-root)
            if (( $# == 0 )); then usage >&2; exit 1; fi
            npu_model_root="$1"
            shift
            ;;
        -*) printf 'Unknown option: %s\n' "$argument" >&2; exit 1 ;;
        *)
            if [[ -n "$model" ]]; then usage >&2; exit 1; fi
            model="$argument"
            ;;
    esac
done
if [[ "$output_type" != int8 && "$output_type" != float32 ]]; then
    printf 'Output type must be int8 or float32.\n' >&2; exit 1
fi
if [[ ! -d "$sample_path" || ! -f "$labels" ||
      ! -x "$python" || ! -x "$cmake" ]]; then
    usage >&2
    printf 'Source model, image directory, labels or project environment missing.\n' >&2
    exit 1
fi
if [[ -n "$npu_model_root" ]]; then
    if [[ -n "$model" || ! -d "$npu_model_root" ]]; then
        printf 'Use either a source model or --npu-model-root with an existing directory.\n' >&2
        exit 1
    fi
elif [[ ! -f "$model" ]]; then
    printf 'Supply an existing source model or --npu-model-root.\n' >&2
    exit 1
fi
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ || ! "$fvp_timeout" =~ ^[1-9][0-9]*$ ]]; then
    printf 'BUILD_JOBS and FVP_TIMEOUT must be positive integers.\n' >&2
    exit 1
fi
if (( run_fvp && ! dry_run && ! models_only )) && [[ ! -x "$fvp" ]]; then
    printf 'FVP executable missing: %s\n' "$fvp" >&2; exit 1
fi
unset PYTHONHOME
export PATH="$venv_path/bin:$PATH"
export PYTHONPATH="$repo_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
# Validate the actual source rather than relying on an unrelated export report.
sample_count="$("$python" - "$model" "$sample_path" <<'PY'
from pathlib import Path
import sys
from PIL import Image
from ethosu.vela.tflite.Model import Model
from ethosu.vela.tflite.TensorType import TensorType
from qrcode_tflite_utils import inspect_tflite

source, samples = map(Path, sys.argv[1:])
if source.is_file():
    inspection = inspect_tflite(source)
    if inspection["npu_delegate_count"]:
        raise SystemExit("For compiled NPU models use --npu-model-root with all 6 variants")
    graph = Model.GetRootAsModel(source.read_bytes(), 0).Subgraphs(0)
    shapes = [[1, 32, 32, 47], [1, 16, 16, 47], [1, 8, 8, 47], [1, 1, 1344]]
    inputs = [graph.Tensors(graph.Inputs(i)) for i in range(graph.InputsLength())]
    outputs = [graph.Tensors(graph.Outputs(i)) for i in range(graph.OutputsLength())]
    if (len(inputs) != 1 or list(inputs[0].ShapeAsNumpy()) != [1, 256, 256, 1]
            or sorted(tuple(t.ShapeAsNumpy()) for t in outputs) != sorted(map(tuple, shapes))
            or any(t.Type() != TensorType.INT8 for t in inputs + outputs)):
        raise SystemExit("Source must have the qrcode NHWC INT8 input/output contract")
images = []
stems = set()
for path in sorted(samples.rglob("*.*")):
    if not path.is_file():
        continue
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError):
        continue
    stem = path.stem.replace(" ", "_")
    if stem in stems:
        raise SystemExit(f"Duplicate image stem would overwrite generated C: {path}")
    stems.add(stem)
    images.append(path)
if not images:
    raise SystemExit("No readable input images")
print(len(images))
PY
)"
printf 'All %s images will be embedded in each of the 12 firmwares.\n' "$sample_count"
run_logged() {
    local log="$1"
    shift
    if (( dry_run )); then
        printf '%q ' "$@"
        printf '\n'
    else
        mkdir -p -- "$(dirname -- "$log")"
        "$@" 2>&1 | tee "$log"
    fi
}
common=(
    "-DCMAKE_TOOLCHAIN_FILE=$repo_root/scripts/cmake/toolchains/bare-metal-gcc.cmake"
    -DCMAKE_BUILD_TYPE=Release -DTARGET_PLATFORM=mps4 -DTARGET_SUBSYSTEM=sse-320
    "-DMLEK_PYTHON_VENV=$venv_path" "-DRESOURCES_PATH=$repo_root/resources_downloaded"
    -DML_FRAMEWORK=tflm -DUSE_CASE_BUILD=qrcode -DUSE_SINGLE_INPUT=OFF
    -DINTERACTIVE_MODE=OFF -DCPU_PROFILE_ENABLED=ON
    -DETHOS_U_NPU_ENABLED=ON -DETHOS_U_NPU_ID=U85 -DETHOS_U_NPU_CONFIG_ID=Z1024
    -DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON
    "-DTA_CONFIG_FILE=$repo_root/scripts/cmake/timing_adapter/ta_config_u85_sys_dram_mid.cmake"
    "-Dqrcode_FILE_PATH=$sample_path" "-Dqrcode_LABELS_FILE=$labels"
    -Dqrcode_ACTIVATION_BUF_SZ=0x00400000
)
declare -A model_paths
for variant in shared 64 128 256 384 768; do
    mode=Dedicated_Sram
    cache_bytes=0
    if [[ "$variant" == shared ]]; then
        mode=Shared_Sram
    else
        cache_bytes=$((variant * 1024))
    fi
    if [[ -n "$npu_model_root" ]]; then
        candidates=()
        for candidate in "$npu_model_root/$variant/"*_vela.tflite; do
            if [[ -f "$candidate" && "$candidate" != *_float_outputs_vela.tflite ]]; then
                candidates+=("$candidate")
            fi
        done
        if (( ${#candidates[@]} != 1 )) || [[ ! -f "${candidates[0]}" ]]; then
            printf 'Expected exactly one NPU model in %s/%s/\n' "$npu_model_root" "$variant" >&2
            exit 1
        fi
        compiled="${candidates[0]}"
    else
        model_dir="$output/models/$variant"
        compiled="$model_dir/$(basename -- "${model%.tflite}")_vela.tflite"
        vela=("$python" -m ethosu.vela "$model"
            --config "$repo_root/scripts/vela/default_vela.ini"
            --accelerator-config ethos-u85-1024 --system-config Ethos_U85_SYS_DRAM_Mid_1024
            --memory-mode "$mode" --optimise Performance --cop-format COP1
            --verbose-performance --output-dir "$model_dir")
        if [[ "$mode" == Dedicated_Sram ]]; then
            vela+=(--arena-cache-size "$cache_bytes")
        fi
        run_logged "$model_dir/compile.log" "${vela[@]}"
    fi
    if (( ! dry_run )) || [[ -n "$npu_model_root" ]]; then
        "$python" - "$compiled" "$mode" "$cache_bytes" <<'CHECK'
import csv
from pathlib import Path
import sys
from ethosu.vela.tflite.Model import Model
from qrcode_tflite_utils import inspect_tflite
from mlek_tools.validation.npu_validation import NpuValidationArgs
from mlek_tools.validation.tflite_validator import validate_tflite_model

path, mode, cache = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
inspection = inspect_tflite(path, require_npu=True)
validate_tflite_model(path, NpuValidationArgs(memory_mode=mode, config="ethos-u85-1024"))
graph = Model.GetRootAsModel(path.read_bytes(), 0).Subgraphs(0)
inputs = [graph.Tensors(graph.Inputs(i)) for i in range(graph.InputsLength())]
outputs = [graph.Tensors(graph.Outputs(i)) for i in range(graph.OutputsLength())]
expected = [(1, 32, 32, 47), (1, 16, 16, 47), (1, 8, 8, 47), (1, 1, 1344)]
if (len(inputs) != 1 or list(inputs[0].ShapeAsNumpy()) != [1, 256, 256, 1]
        or sorted(tuple(t.ShapeAsNumpy()) for t in outputs) != sorted(expected)):
    raise SystemExit(f"NPU model has wrong qrcode interface: {path}")
summary = path.with_name(path.stem.removesuffix("_vela")
                         + "_summary_Ethos_U85_SYS_DRAM_Mid_1024.csv")
with summary.open(encoding="utf-8") as stream:
    row = next(csv.DictReader(stream))
if (row["memory_mode"] != mode
        or row["accelerator_configuration"] != "Ethos_U85_1024"
        or row["system_config"] != "Ethos_U85_SYS_DRAM_Mid_1024"
        or (mode == "Dedicated_Sram" and int(float(row["arena_cache_size"]) * 1024) != cache)):
    raise SystemExit(f"Vela configuration mismatch: {summary}")
print(f"NPU CHECK PASS: {path} / {mode} / cache={cache}: {inspection}")
CHECK
    fi
    if [[ "$output_type" == float32 ]]; then
        float_model="$output/models/$variant/$(basename -- "${compiled%_vela.tflite}")_float_outputs_vela.tflite"
        run_logged "$output/models/$variant/dequantize.log" \
            "$python" "$repo_root/scripts/qrcode_output_io.py" \
            --model "$compiled" --output "$float_model"
        compiled="$float_model"
    fi
    model_paths[$variant]="$compiled"
done
if (( dry_run )); then
    printf '\nDRY RUN: no models compiled, firmware built, or FVP executed.\n'
else
    printf '\nAll 6 NPU models checked before firmware builds.\n'
fi
if (( models_only )); then exit 0; fi
for variant in shared 64 128 256 384 768; do
    mode=Dedicated_Sram
    cache_bytes=0
    if [[ "$variant" == shared ]]; then
        mode=Shared_Sram
    else
        cache_bytes=$((variant * 1024))
    fi
    compiled="${model_paths[$variant]}"
    for deployment in fpga fvp; do
        build_dir="$output/$deployment/$variant"
        fpga=OFF
        framebuffer=0x96000000
        if [[ "$deployment" == fpga ]]; then
            fpga=ON
            framebuffer=0x74000000
        fi
        printf '\nConfiguration: %s / %s / cache %s bytes\n' "$deployment" "$mode" "$cache_bytes"
        run_logged "$build_dir/configure.log" "$cmake" --fresh -S "$repo_root" \
            -B "$build_dir" -G 'Unix Makefiles' "${common[@]}" \
            "-DFPGA_PLATFORM_SSE_320=$fpga" "-DETHOS_U_NPU_MEMORY_MODE=$mode" \
            "-DETHOS_U_NPU_CACHE_SIZE=$cache_bytes" "-Dqrcode_MODEL_PATH=$compiled" \
            "-DHDLCD_FRAME_BUFFER_BASE_ADDRESS=$framebuffer"
        run_logged "$build_dir/build.log" "$cmake" --build "$build_dir" \
            --target mlek_qrcode --parallel "$jobs"
        if [[ "$deployment" == fvp ]] && (( run_fvp )); then
            # Confine vendor PYTHONHOME and library settings to this FVP run.
            (
                if (( ! dry_run )); then
                    vendor_runtime="$(dirname -- "$fvp")/../../scripts/runtime.sh"
                    if [[ -f "$vendor_runtime" ]]; then
                        set +eu
                        source "$vendor_runtime"
                        set -eu
                    fi
                fi
                printf 'FVP log: %s/fvp.log\n' "$build_dir"
                run_logged "$build_dir/fvp.log" timeout --signal=TERM "$fvp_timeout" "$fvp" \
                    -a "$build_dir/bin/mlek_qrcode.axf" \
                    -C mps4_board.subsystem.ethosu.num_macs=1024 \
                    -C mps4_board.telnetterminal0.start_telnet=0 \
                    -C mps4_board.uart0.out_file=- -C mps4_board.uart0.shutdown_on_eot=1 \
                    -C mps4_board.visualisation.disable-visualisation=0 \
                    -C vis_hdlcd.disable_visualisation=0 --stat
            )
            if (( ! dry_run )); then
                if ! rg -q 'Main loop terminated successfully' "$build_dir/fvp.log" ||
                   rg -qi 'terminated with failure|Failed to initialise|readback mismatch' "$build_dir/fvp.log"; then
                    printf 'FVP validation failed: %s/fvp.log\n' "$build_dir" >&2
                    exit 1
                fi
                if [[ "$output_type" == float32 ]]; then
                    dequantized="$(rg -c 'Profile for OutputDequantization:' "$build_dir/fvp.log" || true)"
                    if [[ "$dequantized" != "$sample_count" ]]; then
                        printf 'Expected %s output-dequantization profiles, found %s in %s/fvp.log\n' \
                            "$sample_count" "${dequantized:-0}" "$build_dir" >&2
                        exit 1
                    fi
                fi
                processed="$(rg -c 'Running image [0-9]+:' "$build_dir/fvp.log" || true)"
                if [[ "$processed" != "$sample_count" ]]; then
                    printf 'Expected %s images, found %s in %s/fvp.log\n' \
                        "$sample_count" "${processed:-0}" "$build_dir" >&2
                    exit 1
                fi
            fi
        fi
    done
done
if (( dry_run )); then
    printf '\nCommand preview finished; no firmware was built or run.\n'
else
    printf '\n12 firmware builds completed. Output directory: %s\n' "$output"
fi
