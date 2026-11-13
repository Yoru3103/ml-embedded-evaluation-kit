#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${IMG_CLASS_BUILD_DIR:-$repo_root/build-img-class-vsi}"
fvp_path="${FVP_PATH:-$HOME/FVP_Corstone_SSE-320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320}"
runtime_path="$(dirname -- "$fvp_path")/../../scripts/runtime.sh"
firmware="$build_dir/bin/mlek_img_class.axf"
vsi_path="${VSI_PYTHON_PATH:-$repo_root/scripts/vsi/python}"

if [[ ! -x "$fvp_path" || ! -f "$runtime_path" || ! -f "$firmware" ]]; then
    printf 'FVP/runtime or image-classification firmware is missing: %s\n' "$firmware" >&2
    exit 1
fi
if [[ ! -f "$vsi_path/arm_vsi4.py" || ! -f "$vsi_path/vsi_video.py" ]]; then
    printf 'Windows VSI client scripts are missing: %s\n' "$vsi_path" >&2
    exit 1
fi

# On mirrored WSL networking, set VSI_VIDEO_HOST=127.0.0.1 explicitly.
if [[ -z "${VSI_VIDEO_HOST:-}" ]]; then
    VSI_VIDEO_HOST="$(ip -4 route show default | awk 'NR == 1 {print $3}')"
fi
if [[ -z "$VSI_VIDEO_HOST" ]]; then
    printf 'Set VSI_VIDEO_HOST to the Windows camera server address.\n' >&2
    exit 1
fi
export VSI_VIDEO_HOST
printf 'Image classification: KEY0 capture; Windows VSI server: %s:6000\n' "$VSI_VIDEO_HOST"

# Vendor runtime expects an interactive shell without nounset/errexit.
set +eu
source "$runtime_path"
set -eu

exec "$fvp_path" -a "$firmware" \
    -C mps4_board.subsystem.ethosu.num_macs=1024 \
    -C "mps4_board.v_path=$vsi_path" \
    -C mps4_board.telnetterminal0.start_telnet=0 \
    -C mps4_board.uart0.out_file=- \
    -C mps4_board.uart0.shutdown_on_eot=1 \
    -C mps4_board.visualisation.disable-visualisation=0 \
    -C vis_hdlcd.disable_visualisation=0 "$@"
