#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
# Run from Linux/WSL. Keep this process open while the host sends WAV requests.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-$repo_root/build/asr-uart/dedicated-256k-fvp}"
port="${2:-5000}"
fvp="${FVP_PATH:-$HOME/FVP_Corstone_SSE-320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320}"
if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65532)); then
    echo 'Choose a TCP start port between 1 and 65532 (four UART ports are opened).' >&2
    exit 1
fi
cache="$build_dir/CMakeCache.txt"
for setting in 'asr_INPUT_MODE=UART' 'FPGA_PLATFORM_SSE_320=OFF' \
    'SEMIHOSTING_ENABLED=OFF' 'ETHOS_U_NPU_TIMING_ADAPTER_ENABLED=OFF'; do
    if ! grep -qE "^${setting%=*}:[^=]+=${setting#*=}[[:space:]]*$" "$cache"; then
        echo "Expected FVP UART fast-mode build setting: $setting" >&2
        exit 1
    fi
done
if [[ ! -f "$build_dir/bin/mlek_asr.axf" || ! -x "$fvp" ]]; then
    echo 'Missing FVP executable or mlek_asr.axf; build with -Target FVP first.' >&2
    exit 1
fi
runtime="$(dirname "$fvp")/../../scripts/runtime.sh"
if [[ -f "$runtime" ]]; then
    # The vendor runtime probes unset variables and unsuccessful grep results.
    set +eu
    source "$runtime"
    set -eu
fi
echo "Functional fast mode: simulated processing times are not FPGA performance measurements."
echo "Connect to the UART0 listening port printed by FVP (requested $port)."
exec "$fvp" -a "$build_dir/bin/mlek_asr.axf" \
    -C mps4_board.subsystem.ethosu.num_macs=1024 \
    -C 'mps4_board.subsystem.ethosu.extra_args=--fast' \
    -C mps4_board.telnetterminal0.mode=raw \
    -C "mps4_board.telnetterminal0.start_port=$port" \
    -C mps4_board.telnetterminal0.start_telnet=0 \
    -C mps4_board.telnetterminal1.start_telnet=0 \
    -C mps4_board.telnetterminal2.start_telnet=0 \
    -C mps4_board.telnetterminal5.start_telnet=0 \
    -C mps4_board.uart0.shutdown_on_eot=0 \
    -C mps4_board.uart0.out_file=- \
    -C mps4_board.visualisation.disable-visualisation=1 \
    -C vis_hdlcd.disable_visualisation=0
