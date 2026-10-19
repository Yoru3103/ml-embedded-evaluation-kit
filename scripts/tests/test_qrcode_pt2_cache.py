# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Exercise one calibration reused across actual Ethos-U memory variants."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from export_model import run_export
from test_qrcode_pt_export import fixture_args, make_checkpoint

from model_export.qrcode.pt import prepare_pte_source
from model_export.qrcode.backends import quantize, quantize_pte, torch_predict
from model_export.qrcode.pt2_cache import load_cache


class Pt2CacheTests(unittest.TestCase):
    """Check cached quantization, backend compilation and stale-cache rejection."""

    def test_shared_and_dedicated_reuse_one_calibration(self):
        """Compile two real PTEs while forbidding calibration after the common export."""
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = fixture_args(directory)
            args.pt2_only = True
            with patch("ultralytics.YOLO", return_value=make_checkpoint()), \
                 patch("model_export.qrcode.backends.quantize", wraps=quantize) as calibrate:
                report = run_export(args)
            self.assertEqual(calibrate.call_count, 1)
            self.assertEqual(report["status"], "completed")
            common = args.output_dir
            self.assertTrue((common / "quantized.pt2").is_file())
            self.assertFalse(list(common.rglob("*.pte")))
            with patch("ultralytics.YOLO", side_effect=AssertionError("PT loaded again")), \
                 patch("model_export.qrcode.backends.quantize",
                       side_effect=AssertionError("Recalibrated")):
                run_export(args)
                args.pt2_only = False
                args.quantized_pt2 = common / "quantized.pt2"
                args.memory_mode = "all"
                args.arena_cache_size = 65536
                args.output_dir = directory / "variants"
                result = run_export(args)
                for mode in ("Shared_Sram", "Dedicated_Sram"):
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual(result["pte"][mode]["cpu_operator_count"], 0)
                    self.assertGreater(result["pte"][mode]["npu_delegate_count"], 0)
            args.per_channel = not args.per_channel
            with self.assertRaisesRegex(ValueError, "cache mismatch"):
                load_cache(common / "quantized.pt2", args)

    def test_direct_and_cached_exports_preserve_quantization(self):
        """Keep integer I/O parameters and contracts identical across PT2 serialization."""
        for output_type in ("int8", "float32"):
            with self.subTest(output_type=output_type), \
                 TemporaryDirectory() as temporary:
                directory = Path(temporary)
                args = fixture_args(directory)
                args.output_type = output_type
                with patch("ultralytics.YOLO", return_value=make_checkpoint()):
                    direct = run_export(args)["pte"]["Dedicated_Sram"]
                    args.output_dir = directory / "common"
                    args.pt2_only = True
                    run_export(args)
                args.quantized_pt2 = args.output_dir / "quantized.pt2"
                args.pt2_only = False
                args.output_dir = directory / "cached"
                with patch("ultralytics.YOLO", side_effect=AssertionError("PT loaded again")), \
                     patch("model_export.qrcode.backends.quantize",
                           side_effect=AssertionError("Recalibrated")):
                    cached = run_export(args)["pte"]["Dedicated_Sram"]
                for key in ("io_quantization", "serialized_tensor_layouts", "execution",
                            "validation"):
                    self.assertEqual(direct[key], cached[key], key)
                self.assertTrue(Path(cached["path"]).is_file())

    def test_nchw_cache_preserves_quantized_outputs(self):
        """Preserve NCHW calibration values independently of compiled layout constraints."""
        with TemporaryDirectory() as temporary:
            args = fixture_args(Path(temporary))
            args.io_layout = "nchw"
            args.save_intermediate = False
            with patch("ultralytics.YOLO", return_value=make_checkpoint()):
                model, samples, source = prepare_pte_source(args, None)
                direct = quantize_pte(model, [Path(path) for path in source["calibration_images"]],
                                      source["contract"], args)
                args.pt2_only = True
                run_export(args)
            cached = torch.export.load(args.output_dir / "quantized.pt2").module(check_guards=False)
            for sample in samples:
                expected = torch_predict(direct, sample["input"], "nchw")
                actual = torch_predict(cached, sample["input"], "nchw")
                for reference, observed in zip(expected, actual):
                    np.testing.assert_array_equal(reference, observed)


if __name__ == "__main__":
    unittest.main()
