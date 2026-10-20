# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Persist completed and failed export stages and per-model reports."""

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path

def write_manifest(path: Path, report: dict):
    """Persist stage status so failed exports cannot look like completed runs.

    :param path:    JSON report destination.
    :param report:  Configuration and completed stages.
    """
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


@contextmanager
def export_manifest(path: Path, report: dict) -> Iterator[dict]:
    """Record completion or failure for source-specific export stages.

    :param path:    Destination JSON manifest.
    :param report:  Mutable report updated by the export stages.
    :returns:       Context yielding the mutable report.
    """
    write_manifest(path, report)
    try:
        yield report
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    else:
        report["status"] = "completed"
    finally:
        write_manifest(path, report)


def save_report(output: Path, report: dict):
    """Write a human-readable sidecar with the contract, configuration and diagnostics."""
    path = output.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {path}", flush=True)
