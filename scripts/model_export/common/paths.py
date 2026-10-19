# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Locate repository resources independently of the working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
