#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

# Usage: bash scripts/build_qrcode_fastcache_sse320.sh [all|fvp|fpga] [all|tflm|pte] [all|64|128|256|384|768]
# Run compile_qrcode_fastcache.py first. Each model's Vela and firmware cache sizes must match.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# > 3 )); then
    printf 'Usage: bash %s [all|fvp|fpga] [all|tflm|pte] [all|64|128|256|384|768]\n' \
        "${BASH_SOURCE[0]}" >&2
    exit 1
fi
case "${1:-all}" in
    all) deployments=(fvp fpga) ;;
    fvp|fpga) deployments=("$1") ;;
    *) printf 'Deployment must be all, fvp or fpga.\n' >&2; exit 1 ;;
esac
case "${2:-all}" in
    all) frameworks=(tflm pte) ;;
    tflm|pte) frameworks=("$2") ;;
    *) printf 'Framework must be all, tflm or pte.\n' >&2; exit 1 ;;
esac
case "${3:-all}" in
    all) cache_sizes=(64 128 256 384 768) ;;
    64|128|256|384|768) cache_sizes=("$3") ;;
    *) printf 'Cache size must be all, 64, 128, 256, 384 or 768 KiB.\n' >&2; exit 1 ;;
esac
for deployment in "${deployments[@]}"; do
    for framework in "${frameworks[@]}"; do
        for cache_kib in "${cache_sizes[@]}"; do
            QRCODE_FASTCACHE_KIB="$cache_kib" \
                bash "$repo_root/scripts/build_qrcode_sse320.sh" \
                "$deployment" "$framework-dedicated"
        done
    done
done
