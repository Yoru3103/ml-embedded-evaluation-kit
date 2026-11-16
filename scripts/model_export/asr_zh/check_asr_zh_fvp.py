# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Check FVP completion, operator counts and full output tensors against INT8 PT2."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def compare_dump(path: Path, reference: object, chunk: int) -> list:
    """Compare all 75 concatenated outputs, preserving the integer counter."""
    data = path.read_bytes()
    offset = 0
    result = []
    for slot in range(75):
        expected = reference[f"chunk{chunk}_output{slot}"]
        count = expected.nbytes
        assert offset + count <= len(data), "Truncated FVP output dump"
        actual = np.frombuffer(data[offset:offset + count], dtype=expected.dtype).reshape(
            expected.shape)
        assert np.isfinite(actual).all(), f"Non-finite output {slot}"
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        result.append({"slot": slot, "shape": list(expected.shape),
                       "max_abs_error": float(error.max()), "mean_abs_error": float(error.mean()),
                       "within_smoke_tolerance": bool(
                           error.mean() <= max(0.05, 0.1 * np.abs(expected).mean())),
                       "relative_l2_error": float(np.linalg.norm(error) /
                                                  max(np.linalg.norm(expected), 1e-12))})
        if slot == 74:
            assert np.array_equal(actual, expected), "Processed frame counter mismatch"
        offset += count
    assert offset == len(data), "Unexpected trailing FVP output bytes"
    return result


def check_log(path: Path, fixed: bool) -> dict:
    """Check successful application completion, per-chunk calls and transcript."""
    text = path.read_text(errors="replace")
    assert "ASR_ZH completed success=1" in text, "Application did not complete successfully"
    assert "CALL_DELEGATE execute failed" not in text
    assert "ASR_ZH_TEXT_REFERENCE match=0" not in text
    if not fixed:
        assert "ASR_ZH_TEXT_REFERENCE match=1" in text, "Missing end-to-end text match"
    paths = re.findall(r"ASR_ZH_PATH chunk=(\d+) ([0-9 ]+)", text)
    operations = re.findall(
        r"(?:quantize_calls|calls quantize)=(\d+).*? (?:npu_calls|npu)=(\d+).*? "
        r"(?:dequantize_calls|dequantize)=(\d+)", text)
    assert paths and len(operations) == len(paths)
    assert all(tuple(map(int, counts)) == (74, 2, 74) for counts in operations)
    return {"chunks": len(paths), "paths": [list(map(int, row.split())) for _, row in paths],
            "successful": True, "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    """Write a numerical report; comparisons are evidence rather than exact NPU equivalence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("build_directory", type=Path)
    parser.add_argument("--fixed", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    result = check_log(args.log, args.fixed)
    assets = json.loads((repo / "resources/asr_zh/validation.json").read_text())
    expected = assets["path"][:result["chunks"]]
    assert len(expected) == result["chunks"]
    result["frame_argmax_agreement"] = sum(
        a == b for x, y in zip(result["paths"], expected) for a, b in zip(x, y)
    ) / (16 * result["chunks"])
    assert result["frame_argmax_agreement"] >= 0.95, "Low frame agreement in smoke test"
    with np.load(repo / "resources_downloaded/asr_zh/dedicated-768k/pt2_reference.npz") as data:
        result["tensor_comparisons"] = [
            compare_dump(
                next(path for path in [
                    args.build_directory / f"asr_zh_000_{chunk:03}_outputs.bin",
                    args.build_directory / f"asr_zh_outputs_{chunk:03}.bin"] if path.exists()),
                data, chunk)
            for chunk in range(min(3, result["chunks"]))]
    result["pte_sha256"] = assets["pte_sha256"]
    result["all_outputs_within_smoke_tolerance"] = all(
        row["within_smoke_tolerance"] for chunk in result["tensor_comparisons"] for row in chunk)
    result["int8_pt2_sha256"] = assets["int8_pt2_sha256"]
    result["limits"] = "FVP functional fast mode; no measured FPGA/silicon performance."
    report = args.log.with_suffix(".json")
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ["paths", "tensor_comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
