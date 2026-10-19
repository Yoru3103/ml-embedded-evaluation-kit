#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

# Usage: bash scripts/qrcode/run_qrcode_fastcache_fvp320.sh [all|tflm|pte] [all|64|128|256|384|768]
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$repo_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
if (( $# > 2 )); then
    printf 'Usage: bash %s [all|tflm|pte] [all|64|128|256|384|768]\n' \
        "${BASH_SOURCE[0]}" >&2
    exit 1
fi
case "${1:-all}" in
    all) frameworks=(tflm pte) ;;
    tflm|pte) frameworks=("$1") ;;
    *) printf 'Framework must be all, tflm or pte.\n' >&2; exit 1 ;;
esac
case "${2:-all}" in
    all) cache_sizes=(64 128 256 384 768) ;;
    64|128|256|384|768) cache_sizes=("$2") ;;
    *) printf 'Cache size must be all, 64, 128, 256, 384 or 768 KiB.\n' >&2; exit 1 ;;
esac
for framework in "${frameworks[@]}"; do
    for cache_kib in "${cache_sizes[@]}"; do
        QRCODE_FASTCACHE_KIB="$cache_kib" \
            bash "$repo_root/scripts/qrcode/run_qrcode_fvp320.sh" "$framework-dedicated"
    done
done
