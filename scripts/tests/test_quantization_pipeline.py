# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Verify that ONNX memory variants also use one real calibration."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import onnx
from PIL import Image

from export_model import run_export
from test_qrcode_contract import make_obb_graph
from model_export.common.cli import parse_args
from model_export.common.paths import ROOT
from model_export.qrcode.backends import quantize


class OnnxQuantizationPipelineTests(unittest.TestCase):
    """Check the source adapter that shares calibration with native PT export."""

    def test_onnx_variants_calibrate_once_and_compile_independent_graphs(self):
        """Run real ONNX adaptation and PT2E while observing variant isolation."""
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = directory / "network.onnx"
            onnx.save(make_obb_graph(3, 8, 15), model)
            images = directory / "images"
            images.mkdir()
            Image.new("RGB", (32, 32), "white").save(images / "sample.png")
            args = parse_args([
                "--model", str(model), "--to", "pte", "--memory-mode", "all",
                "--calibration-dir", str(images), "--calibration-limit", "1",
                "--validation-limit", "1", "--no-save-intermediate",
                "--vela-config", str(ROOT / "scripts/vela/default_vela.ini"),
                "--output-dir", str(directory / "export")])
            with patch("model_export.qrcode.backends.quantize", wraps=quantize) as calibrate, \
                 patch("model_export.qrcode.backends.compile_quantized_pte",
                       return_value={"fixture": True}) as compile_variant:
                report = run_export(args)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(calibrate.call_count, 1)
            self.assertEqual(compile_variant.call_count, 2)
            first, second = [call.args[0] for call in compile_variant.call_args_list]
            self.assertIsNot(first, second)
            self.assertIsNot(first.graph, second.graph)
            self.assertEqual(set(report["pte"]), {"Dedicated_Sram", "Shared_Sram"})


if __name__ == "__main__":
    unittest.main()
