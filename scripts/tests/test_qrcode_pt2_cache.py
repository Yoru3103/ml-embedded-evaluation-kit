# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Exercise one calibration reused across actual Ethos-U memory variants."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from export_model import run_export
from qrcode_pt2_cache import load_cache, quantize
from test_qrcode_pt_export import fixture_args, make_checkpoint


class Pt2CacheTests(unittest.TestCase):
    """Check cached quantization, backend compilation and stale-cache rejection."""

    def test_shared_and_dedicated_reuse_one_calibration(self):
        """Compile two real PTEs while forbidding calibration after the common export."""
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = fixture_args(directory)
            args.pt2_only = True
            with patch("ultralytics.YOLO", return_value=make_checkpoint()), \
                 patch("qrcode_pt2_cache.quantize", wraps=quantize) as calibrate:
                report = run_export(args)
            self.assertEqual(calibrate.call_count, 1)
            self.assertEqual(report["status"], "completed")
            common = args.output_dir
            self.assertTrue((common / "quantized.pt2").is_file())
            self.assertFalse(list(common.rglob("*.pte")))
            with patch("ultralytics.YOLO", side_effect=AssertionError("PT loaded again")), \
                 patch("qrcode_pt2_cache.quantize", side_effect=AssertionError("Recalibrated")):
                run_export(args)
                args.pt2_only = False
                args.quantized_pt2 = common / "quantized.pt2"
                for mode in ("Shared_Sram", "Dedicated_Sram"):
                    args.memory_mode = mode
                    args.arena_cache_size = 65536
                    args.output_dir = directory / mode
                    result = run_export(args)
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual(result["pte"][mode]["cpu_operator_count"], 0)
                    self.assertGreater(result["pte"][mode]["npu_delegate_count"], 0)
            args.per_channel = not args.per_channel
            with self.assertRaisesRegex(ValueError, "cache mismatch"):
                load_cache(common / "quantized.pt2", args)


if __name__ == "__main__":
    unittest.main()
