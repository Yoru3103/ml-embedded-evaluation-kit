# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Check streaming ASR timing partitions and extract a transcript from target logs."""
import argparse
import hashlib
import json
from pathlib import Path
import re

CHUNK_STAGES = ["Preprocess", "Quantize", "NPU.Driver", "Dequantize", "Invoke.Other",
                "Postprocess", "State.Copy", "Invoke.Total", "Pipeline.Total"]
SUMMARY_STAGES = ["Clip.Setup", *CHUNK_STAGES[:7], "Decode.Final", "Invoke.Total",
                  "Pipeline.Total"]


def read_stages(body: str, frequency: int, summary: bool) -> dict:
    """Check stage order and printed units for one measured section."""
    rows = re.findall(r"ASR_ZH_STAGE (\S+)\s+cycles=(\d+) us=(\d+)\.(\d{3})", body)
    assert [row[0] for row in rows] == (SUMMARY_STAGES if summary else CHUNK_STAGES)
    stages = {}
    for name, cycles, whole, fractional in rows:
        count = int(cycles)
        nanoseconds = (count // frequency * 1000000000 +
                       count % frequency * 1000000000 // frequency)
        assert int(whole) * 1000 + int(fractional) == nanoseconds, "Incorrect time conversion"
        stages[name] = {"cycles": count, "microseconds": nanoseconds / 1000}
    return stages


def read_sections(text: str, expected_calls: tuple[int, int, int] = (74, 2, 74)) -> list[dict]:
    """Parse ordered stages and verify the cycle-to-time conversion.

    :param text:    Captured target output.
    :returns:       Per-chunk sections followed by the clip summary.
    """
    headers = list(re.finditer(
        r"ASR_ZH timing: (?:chunk=(\d+)/(\d+)|summary chunks=(\d+)) cpu_hz=(\d+)", text))
    sections = []
    for index, header in enumerate(headers):
        body = text[header.end():headers[index + 1].start() if index + 1 < len(headers)
                    else len(text)]
        frequency = int(header[4])
        assert frequency > 0, "Invalid CPU frequency"
        counts = re.search(r"ASR_ZH_STAGE calls quantize=(\d+) npu=(\d+) dequantize=(\d+)", body)
        if not header[3]:
            assert counts and tuple(map(int, counts.groups())) == expected_calls, (
                "Hook count mismatch")
        sections.append({"chunk": int(header[1]) if header[1] else None,
                         "chunks": int(header[2] or header[3]), "cpu_hz": frequency,
                         "stages": read_stages(body, frequency, bool(header[3]))})
    return sections


def check_partitions(sections: list[dict]):
    """Verify non-overlapping invoke partitions and all clip/chunk sums.

    :param sections:    Parsed timing sections.
    """
    assert len(sections) >= 2 and sections[-1]["chunk"] is None, "Missing summary"
    chunks, summary = sections[:-1], sections[-1]
    assert [row["chunk"] for row in chunks] == list(range(1, len(chunks) + 1))
    assert all(row["chunks"] == len(chunks) for row in sections)
    assert all(row["cpu_hz"] == summary["cpu_hz"] for row in chunks)
    for section in sections:
        stages = {key: value["cycles"] for key, value in section["stages"].items()}
        assert stages["Invoke.Total"] == sum(
            stages[key] for key in ["Quantize", "NPU.Driver", "Dequantize", "Invoke.Other"])
        assert stages["Pipeline.Total"] == sum(
            stages[key] for key in ["Preprocess", "Invoke.Total", "Postprocess", "State.Copy"]
        ) + stages.get("Clip.Setup", 0)
    for name in CHUNK_STAGES[:-1]:
        expected = sum(row["stages"][name]["cycles"] for row in chunks)
        if name == "Postprocess":
            expected += summary["stages"]["Decode.Final"]["cycles"]
        assert summary["stages"][name]["cycles"] == expected, name


def main():
    """Write a report without treating FVP fast-mode timing as board performance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--int8-state", action="store_true")
    args = parser.parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace")
    assert "ASR_ZH completed success=1" in text, "Application failed or did not complete"
    sections = read_sections(text, (1, 2, 1) if args.int8_state else (74, 2, 74))
    check_partitions(sections)
    result = re.search(r"ASR_ZH_TEXT clip=0 chunks=(\d+) text=([^\r\n]*)", text)
    total = re.search(r"ASR_ZH_TOTAL samples=(\d+) cycles=(\d+) text_cycles=(\d+) "
                      r"computed_feature_frames=(\d+)", text)
    assert result and total and int(result[1]) == len(sections) - 1
    assert int(total[2]) == sections[-1]["stages"]["Pipeline.Total"]["cycles"]
    assert int(total[3]) == sections[-1]["stages"]["Decode.Final"]["cycles"]
    report = {"successful": True, "chunks": int(result[1]), "transcript": result[2],
              "samples": int(total[1]), "computed_feature_frames": int(total[4]),
              "hook_counts_and_timing_partitions_verified": True,
              "log_sha256": hashlib.sha256(args.log.read_bytes()).hexdigest(),
              "sections": sections}
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        assert reference["samples"] == report["samples"] and reference["chunks"] == report["chunks"]
        assert report["computed_feature_frames"] == (report["samples"] + 80) // 160
        report.update({"reference_transcript": reference["transcript"],
                       "transcript_matches_int8_pt2": result[2] == reference["transcript"],
                       "audio_sha256": reference["audio_sha256"]})
    output = args.output or args.log.with_suffix(".json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "sections"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
