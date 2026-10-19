#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

# Build 2 platforms x (Shared SRAM + 5 Dedicated SRAM fast-cache sizes).
# Export a native PT grayscale OBB checkpoint to PTE for each memory configuration.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="$repo_root/resources_downloaded/env"
python="$venv_path/bin/python"
cmake="$venv_path/bin/cmake"
sample_path="${QRCODE_SAMPLE_PATH:-$repo_root/resources/qrcode/smaples}"
labels="${QRCODE_LABELS_FILE:-$repo_root/resources/qrcode/labels.txt}"
output="${QRCODE_MATRIX_ROOT:-$repo_root/build-qrcode-pte-matrix}"
jobs="${BUILD_JOBS:-8}"
fvp_timeout="${FVP_TIMEOUT:-1800}"
fvp="${FVP_PATH:-$HOME/FVP_Corstone_SSE-320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320}"
calibration="${QRCODE_CALIBRATION_DIR:-$repo_root/resources_downloaded/qrcode/calibration_merged/images/calibration}"
validation="${QRCODE_VALIDATION_DIR:-$repo_root/resources_downloaded/qrcode/calibration_merged/images/validation}"
threads="${QRCODE_THREADS:-4}"
validation_limit="${QRCODE_VALIDATION_LIMIT:-543}"
max_score_error="${QRCODE_MAX_SCORE_ERROR:-0.9}"
min_box_iou="${QRCODE_MIN_BOX_IOU:-0.1}"
dry_run=0
run_fvp=0
models_only=0
npu_model_root=""
model=""
output_type=int8
quantized_pt2=""
usage() {
    cat <<'HELP'
Usage: bash scripts/build_qrcode_pte_matrix.sh MODEL.pt [--run-fvp] [--dry-run]
   or: bash scripts/build_qrcode_pte_matrix.sh --npu-model-root DIR [--run-fvp]

MODEL must be the 15-class, 8-bin DFL QR-code OBB PT checkpoint.
Calibrates PT once into OUTPUT/common/quantized.pt2, then compiles 6 PTEs.
The common PT2 is reused on later runs when source/configuration/data match.
Builds FPGA and FVP firmware for Shared_Sram and Dedicated_Sram 64/128/256/384/768 KiB.
Every firmware runs all images, without interaction. --run-fvp also executes the 6 FVPs.
FPGA outputs must be loaded onto the board separately.
--models-only exports/checks all 6 PTE models without firmware builds.
--output-type {int8,float32} selects output tensors (default: int8).
float32 retains four terminal CPU dequantize calls; the network stays on Ethos-U.
--quantized-pt2 FILE reuses a checked common cache from a previous matrix run,
without loading the PT checkpoint or repeating calibration.
--npu-model-root reuses DIR/{shared,64,128,256,384,768}/pte_MEMORY_MODE/best.pte
with each variant's export_report.json. All 6 models are checked before builds.

Models are written under OUTPUT/models/VARIANT/.
Build and runtime logs: OUTPUT/{fpga,fvp}/VARIANT/{configure,build,fvp}.log
(fvp.log is generated only for --run-fvp.)

Environment: QRCODE_SAMPLE_PATH, QRCODE_LABELS_FILE, QRCODE_MATRIX_ROOT,
             BUILD_JOBS (8), FVP_PATH, FVP_TIMEOUT (1800 seconds per firmware),
             QRCODE_CALIBRATION_DIR, QRCODE_VALIDATION_DIR, QRCODE_THREADS (4),
             QRCODE_VALIDATION_LIMIT (543), QRCODE_MAX_SCORE_ERROR (0.9),
             QRCODE_MIN_BOX_IOU (0.1). Calibration uses every image; seed is fixed.
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
        --output-type|--quantized-pt2)
            if (( $# == 0 )); then usage >&2; exit 1; fi
            if [[ "$argument" == --output-type ]]; then
                output_type="$1"
            else
                quantized_pt2="$1"
            fi
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
if [[ -n "$quantized_pt2" && ( ! -f "$quantized_pt2" || -n "$npu_model_root" ) ]]; then
    printf -- '--quantized-pt2 requires an existing cache and a PT source, not --npu-model-root.\n' >&2
    exit 1
fi
if [[ ! -d "$sample_path" || ! -f "$labels" ||
      ! -x "$python" || ! -x "$cmake" ]]; then
    usage >&2
    printf 'Source model, image directory, labels or project environment missing.\n' >&2
    exit 1
fi
if [[ -n "$npu_model_root" ]]; then
    if [[ -n "$model" || ! -d "$npu_model_root" ]]; then
        printf 'Use either a PT source or --npu-model-root with an existing directory.\n' >&2
        exit 1
    fi
elif [[ ! -f "$model" || "$model" != *.pt || ! -d "$calibration" || ! -d "$validation" ]]; then
    printf 'Supply a PT checkpoint and existing calibration/validation directories.\n' >&2
    exit 1
fi
if [[ ! "$threads" =~ ^[1-9][0-9]*$ || ! "$validation_limit" =~ ^[1-9][0-9]*$ ]]; then
    printf 'QRCODE_THREADS and QRCODE_VALIDATION_LIMIT must be positive integers.\n' >&2
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
# Count every image that will be embedded in each firmware.
sample_count="$("$python" - "$model" "$sample_path" <<'PY'
from pathlib import Path
import sys
from PIL import Image
source, samples = map(Path, sys.argv[1:])
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
    "-DCMSIS_NN_LOCAL_PATH=$repo_root/dependencies/cmsis-nn"
    -DML_FRAMEWORK=executorch -DUSE_CASE_BUILD=qrcode -DUSE_SINGLE_INPUT=OFF
    -DINTERACTIVE_MODE=OFF -DCPU_PROFILE_ENABLED=ON
    -DETHOS_U_NPU_ENABLED=ON -DETHOS_U_NPU_ID=U85 -DETHOS_U_NPU_CONFIG_ID=Z1024
    -DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON
    "-DTA_CONFIG_FILE=$repo_root/scripts/cmake/timing_adapter/ta_config_u85_sys_dram_mid.cmake"
    "-Dqrcode_FILE_PATH=$sample_path" "-Dqrcode_LABELS_FILE=$labels"
    -Dqrcode_ACTIVATION_BUF_SZ=0x00400000
)
if [[ -z "$npu_model_root" && -z "$quantized_pt2" ]]; then
        run_logged "$output/common/calibration.log" "$python" "$repo_root/scripts/export_model.py" \
        --model "$model" --to pte --io-layout nhwc --image-size 256 --output-type "$output_type" \
        --preprocessing mlek-crop --calibration-dir "$calibration" --calibration-limit 0 \
        --validation-dir "$validation" --validation-limit "$validation_limit" \
        --seed 0 --threads "$threads" --per-channel \
        --max-score-error "$max_score_error" --min-box-iou "$min_box_iou" \
        --target ethos-u85-1024 --vela-config "$repo_root/scripts/vela/default_vela.ini" \
        --system-config Ethos_U85_SYS_DRAM_Mid_1024 --memory-mode Dedicated_Sram \
        --arena-cache-size 393216 \
        --extra-flag=--optimise=Performance --extra-flag=--cop-format=COP1 \
        --extra-flag=--verbose-performance --output-dir "$output/common" --pt2-only
fi
quantized_pt2="${quantized_pt2:-$output/common/quantized.pt2}"
declare -A model_paths
for variant in shared 64 128 256 384 768; do
    mode=Dedicated_Sram
    cache_bytes=0
    if [[ "$variant" == shared ]]; then
        mode=Shared_Sram
    else
        cache_bytes=$((variant * 1024))
    fi
    model_dir="${npu_model_root:-$output/models}/$variant"
    compiled="$model_dir/pte_$mode/best.pte"
    # Shared SRAM does not use the dedicated cache; keep a positive exporter argument.
    export_cache="$cache_bytes"
    if [[ "$mode" == Shared_Sram ]]; then export_cache=393216; fi
    if [[ -z "$npu_model_root" ]]; then
        run_logged "$model_dir/export.log" "$python" "$repo_root/scripts/export_model.py" \
        --model "$model" --to pte --io-layout nhwc --image-size 256 --output-type "$output_type" \
        --preprocessing mlek-crop --calibration-dir "$calibration" --calibration-limit 0 \
        --validation-dir "$validation" --validation-limit "$validation_limit" \
        --seed 0 --threads "$threads" --per-channel \
        --max-score-error "$max_score_error" --min-box-iou "$min_box_iou" \
        --target ethos-u85-1024 --vela-config "$repo_root/scripts/vela/default_vela.ini" \
        --system-config Ethos_U85_SYS_DRAM_Mid_1024 --memory-mode "$mode" \
        --arena-cache-size "$export_cache" \
        --extra-flag=--optimise=Performance --extra-flag=--cop-format=COP1 \
        --extra-flag=--verbose-performance --output-dir "$model_dir" \
        --quantized-pt2 "$quantized_pt2"
    fi
    model_paths[$variant]="$compiled"
    if (( ! dry_run )) || [[ -n "$npu_model_root" ]]; then
        "$python" - "$model_dir/export_report.json" "$mode" "$compiled" "$cache_bytes" "$output_type" <<'CHECK'
import json
from pathlib import Path
import sys
import re
from executorch.exir._serialize import _deserialize_pte_binary
from qrcode_onnx_utils import check_pte_contract
from qrcode_output_io import check_pte_execution

report = json.loads(Path(sys.argv[1]).read_text())
if (report["status"] != "completed"
        or report.get("source_route") != "PT -> native grayscale raw OBB -> PT2E -> Ethos-U PTE"
        or report["contract"]["input"]["shape"] != [1, 1, 256, 256]):
    raise SystemExit("Native grayscale PT export did not complete")
record = report["pte"][sys.argv[2]]
output_type = sys.argv[5]
if record.get("output_type", "int8") != output_type or record["npu_delegate_count"] < 1:
    raise SystemExit("PTE report output type or Ethos-U delegation mismatch")
path, mode, cache = Path(sys.argv[3]), sys.argv[2], int(sys.argv[4])
data = path.read_bytes()
program = _deserialize_pte_binary(data).program
plans = [plan for plan in program.execution_plan if plan.name == "forward"]
if len(plans) != 1:
    raise SystemExit("Expected one forward execution plan")
plan = plans[0]
execution = check_pte_execution(plan, output_type)
if record["cpu_operator_count"] != len(plan.operators):
    raise SystemExit("PTE report CPU operator count differs from the actual execution plan")
check_pte_contract(plan, report["contract"], "nhwc", output_type)
text = data.decode("latin1")
expected = {"memory-mode": mode, "accelerator-config": "ethos-u85-1024",
            "system-config": "Ethos_U85_SYS_DRAM_Mid_1024"}
if mode == "Dedicated_Sram":
    expected["arena-cache-size"] = str(cache)
for flag, value in expected.items():
    actual = re.findall(r"--" + re.escape(flag) + r"=([A-Za-z0-9_-]+)", text)
    if not actual or any(item != value for item in actual):
        raise SystemExit(f"PTE embedded {flag} mismatch: {actual}, expected {value}")
print(f"NPU CHECK PASS: {path} / {mode} / cache={cache}, "
      f"Ethos-U delegates={len(plan.delegates)}, {execution}, "
      f"NHWC INT8 input / {output_type} outputs")
CHECK
    fi
done
if (( dry_run )); then
    printf '\nDRY RUN: no models exported, firmware built, or FVP executed.\n'
else
    printf '\nAll 6 PTE NPU models checked before firmware builds.\n'
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
        memory_options=(-Dqrcode_ET_METHOD_MEMORY_IN_DDR=OFF)
        if [[ "$mode" == Shared_Sram ]]; then
            memory_options=(-Dqrcode_ET_METHOD_MEMORY_IN_DDR=ON
                -DML_FWK_TMP_MEM_BASE=0x31000000 -DML_FWK_TMP_MEM_SIZE=0x00400000)
        fi
        run_logged "$build_dir/configure.log" "$cmake" --fresh -S "$repo_root" \
            -B "$build_dir" -G 'Unix Makefiles' "${common[@]}" "${memory_options[@]}" \
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
                    -C mps4_board.visualisation.disable-visualisation=1 \
                    -C vis_hdlcd.disable_visualisation=1 --stat
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
