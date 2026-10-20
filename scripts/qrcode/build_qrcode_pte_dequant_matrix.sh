#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

# Reuse the complete PT calibration/export/build matrix with four terminal output dequantizers.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$repo_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
for argument in "$@"; do
    if [[ "$argument" == --help || "$argument" == -h ]]; then
        cat <<'HELP'
Usage: bash scripts/qrcode/build_qrcode_pte_dequant_matrix.sh MODEL.pt [OPTIONS]
   or: bash scripts/qrcode/build_qrcode_pte_dequant_matrix.sh --npu-model-root DIR [OPTIONS]

Calibrates the native OBB PT checkpoint once, exports 6 PTEs, then builds 12 firmwares.
Input stays INT8; four terminal dequantizers produce FP32 outputs; the network stays on NPU.
CPU profiling and OutputDequantization timing are ON.
Options: --dry-run, --models-only, --run-fvp, --quantized-pt2 FILE.
--npu-model-root requires 6 PTE variants already exported with FP32 outputs.
--quantized-pt2 reuses an existing common calibration cache with the same source/data/options.
Environment matches build_qrcode_pte_matrix.sh; default QRCODE_MATRIX_ROOT is
build-qrcode-pte-dequant-matrix. Release builds; all sample images are embedded.
HELP
        exit 0
    fi
done
export QRCODE_MATRIX_ROOT="${QRCODE_MATRIX_ROOT:-$repo_root/build-qrcode-pte-dequant-matrix}"
printf 'ExecuTorch dequantization matrix: INT8 input, four FP32 outputs, CPU profiling ON.\n'
exec bash "$repo_root/scripts/qrcode/build_qrcode_pte_matrix.sh" "$@" --output-type float32
