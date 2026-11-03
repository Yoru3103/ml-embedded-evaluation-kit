#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-$repo_root/build/asr_zh/fvp-768k-fixed}"
build_dir="$(cd "$build_dir" && pwd)"
for setting in 'FPGA_PLATFORM_SSE_320=OFF' 'SEMIHOSTING_ENABLED=ON' \
    'USE_CASE_BUILD=asr_zh'; do
    if ! grep -qE "^${setting%=*}:[^=]+=${setting#*=}[[:space:]]*$" "$build_dir/CMakeCache.txt"; then
        echo "Expected FVP configuration: $setting" >&2
        exit 1
    fi
done
cache_value() {
    sed -n "s/^$1:[^=]*=//p" "$build_dir/CMakeCache.txt" | tr -d '\r'
}
mode="$(cache_value ETHOS_U_NPU_MEMORY_MODE)"
cache="$(cache_value ETHOS_U_NPU_CACHE_SIZE)"
case "$mode:$cache" in
    Dedicated_Sram:786432|Shared_Sram:0) ;;
    *) echo "Unsupported ASR memory configuration: $mode, cache=$cache" >&2; exit 1 ;;
esac
input_mode="$(cache_value asr_zh_INPUT_MODE)"
if [[ -n "$input_mode" && "$input_mode" != STATIC ]]; then
    echo "Use the audio demo launcher for $input_mode input." >&2
    exit 1
fi
fvp="${FVP_PATH:-$HOME/FVP_Corstone_SSE-320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320}"
runtime="$(dirname "$fvp")/../../scripts/runtime.sh"
if [[ -f "$runtime" ]]; then
    set +eu
    source "$runtime"
    set -eu
fi
exec "$fvp" -a "$build_dir/bin/mlek_asr_zh.axf" \
    -C mps4_board.subsystem.ethosu.num_macs=1024 \
    -C 'mps4_board.subsystem.ethosu.extra_args=--fast' \
    -C mps4_board.subsystem.cpu0.semihosting-enable=1 \
    -C "mps4_board.subsystem.cpu0.semihosting-cwd=$build_dir" \
    -C mps4_board.telnetterminal0.start_telnet=0 \
    -C mps4_board.telnetterminal1.start_telnet=0 \
    -C mps4_board.telnetterminal2.start_telnet=0 \
    -C mps4_board.telnetterminal5.start_telnet=0 \
    -C mps4_board.uart0.out_file=- \
    -C mps4_board.visualisation.disable-visualisation=1 \
    -C vis_hdlcd.disable_visualisation=1
