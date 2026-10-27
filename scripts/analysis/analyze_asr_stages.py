# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Extract individual ASR timing records, preserving UART order and repeated calls."""
import argparse
import re
from pathlib import Path


def extract(path: Path) -> str:
    """Keep individual timing lines without combining measurements.

    :param path:        UART log file.
    :returns:           Markdown containing the ordered timing lines.
    :raises ValueError: If no profiling records are present.
    """
    lines = path.read_text(encoding="utf8", errors="replace").splitlines()
    pattern = r"(?:Inference \d+/\d+|Profile for .+|CPU TOTAL: .+|DURATION: .+|NPU .+)"
    records = []
    for line in lines:
        match = re.search(pattern, line)
        if match:
            records.append(match.group())
    if not any(line.startswith("Profile for ") for line in records):
        raise ValueError(f"Missing individual profiling records in {path}")
    return f"## {path.name}\n\n```text\n" + "\n".join(records) + "\n```\n"


def main():
    """Extract Dedicated/Shared measurements in their original sequence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [extract(path) for path in args.logs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("# ASR individual stage timings\n\n" + "\n".join(reports),
                           encoding="utf8")
    print(args.output)


if __name__ == "__main__":
    main()
