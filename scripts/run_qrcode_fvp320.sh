#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

# Usage: bash scripts/run_qrcode_fvp320.sh [all|variant]
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
selection="${1:-all}"
fvp_path="${FVP_PATH:-$HOME/FVP_Corstone_SSE-320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320}"
fvp_timeout="${FVP_TIMEOUT:-180}"
if (( $# > 1 )); then
    printf 'Usage: bash %s [all|tflm-shared|tflm-dedicated|pte-shared|pte-dedicated]\n' \
        "${BASH_SOURCE[0]}" >&2
    exit 1
fi
case "$selection" in
    all) variants=(tflm-shared tflm-dedicated pte-dedicated pte-shared) ;;
    tflm-shared|tflm-dedicated|pte-shared|pte-dedicated) variants=("$selection") ;;
    *) printf 'Unsupported variant: %s\n' "$selection" >&2; exit 1 ;;
esac
if [[ ! -x "$fvp_path" || ! "$fvp_timeout" =~ ^[1-9][0-9]*$ ]]; then
    printf 'FVP_PATH must be executable and FVP_TIMEOUT a positive integer.\n' >&2
    exit 1
fi
vendor_runtime="$(dirname -- "$fvp_path")/../../scripts/runtime.sh"
for variant in "${variants[@]}"; do
    if [[ -n "${QRCODE_FASTCACHE_KIB:-}" && "$variant" == *-dedicated ]]; then
        case "$QRCODE_FASTCACHE_KIB" in
            64|128|256|384|768) ;;
            *) printf 'Unsupported fast-cache size.\n' >&2; exit 1 ;;
        esac
        build_dir="$repo_root/build-qrcode-fastcache-fvp-${variant%-dedicated}-${QRCODE_FASTCACHE_KIB}k-z1024-ta-mid"
    else
        build_dir="$repo_root/build-qrcode-fvp-${variant}-z1024-ta-mid"
    fi
    firmware="$build_dir/bin/mlek_yolov8_detection.axf"
    if [[ ! -f "$firmware" ]]; then
        printf 'Firmware is missing: %s\n' "$firmware" >&2
        exit 1
    fi
    printf '\nRunning QR-code FVP / %s\n' "$variant"
    (
        if [[ -f "$vendor_runtime" ]]; then
            # Vendor setup assumes interactive-shell settings; confine them here.
            set +eu
            source "$vendor_runtime"
            set -eu
        fi
        timeout --signal=TERM "$fvp_timeout" "$fvp_path" -a "$firmware" \
            -C mps4_board.subsystem.ethosu.num_macs=1024 \
            -C mps4_board.telnetterminal0.start_telnet=0 \
            -C mps4_board.uart0.out_file=- \
            -C mps4_board.uart0.shutdown_on_eot=1 \
            -C mps4_board.visualisation.disable-visualisation=1 \
            -C vis_hdlcd.disable_visualisation=1 --stat
    ) | tee "$build_dir/fvp.log"
    if ! rg -q 'Main loop terminated successfully' "$build_dir/fvp.log" ||
       rg -q 'terminated with failure|Failed to initialise|readback mismatch' "$build_dir/fvp.log"; then
        printf 'FVP validation failed; inspect %s/fvp.log\n' "$build_dir" >&2
        exit 1
    fi
done
