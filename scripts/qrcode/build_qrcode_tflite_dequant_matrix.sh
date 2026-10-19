#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

# Same arguments and complete build matrix as the INT8 entry, with model-side output dequantization.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$repo_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
for argument in "$@"; do
    if [[ "$argument" == --help || "$argument" == -h ]]; then
        cat <<'HELP'
Usage: bash scripts/qrcode/build_qrcode_tflite_dequant_matrix.sh MODEL.tflite [OPTIONS]
   or: bash scripts/qrcode/build_qrcode_tflite_dequant_matrix.sh --npu-model-root DIR [OPTIONS]

Supply the same uncompiled INT8 TFLite or 6 compiled INT8 variants as the original matrix.
Appends four output DEQUANTIZE operators, validates the models, then builds 12 firmwares.
Input stays INT8; outputs are FP32; CPU profiling and OutputDequantization timing are ON.
Options: --dry-run, --models-only, --run-fvp.
Environment matches build_qrcode_tflite_matrix.sh; default QRCODE_MATRIX_ROOT is
build-qrcode-tflite-dequant-matrix. Release builds; all sample images are embedded.
HELP
        exit 0
    fi
done
export QRCODE_MATRIX_ROOT="${QRCODE_MATRIX_ROOT:-$repo_root/build-qrcode-tflite-dequant-matrix}"
printf 'TFLite dequantization matrix: INT8 input, four FP32 outputs, CPU profiling ON.\n'
exec bash "$repo_root/scripts/qrcode/build_qrcode_tflite_matrix.sh" "$@" --output-type float32
