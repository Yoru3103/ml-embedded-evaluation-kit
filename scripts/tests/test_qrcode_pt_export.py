# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Exercise native OBB PT capture, calibration and Ethos-U serialization."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from export_model import run_export
from export_qrcode_onnx import parse_args
from export_qrcode_pt import ROOT


class ConvLayer(torch.nn.Module):
    """Small routed backbone layer with the checkpoint's first-convolution interface."""

    def __init__(self, channels: int):
        """Create a stride-two convolution.

        :param channels:    Number of input feature channels.
        """
        super().__init__()
        self.f = -1
        self.conv = torch.nn.Conv2d(channels, 4, 3, stride=2, padding=1)
        self.branch = torch.nn.Conv2d(4, 2, 1)
        self.skip = torch.nn.Conv2d(4, 2, 1)
        torch.nn.init.constant_(self.conv.weight, 0.02)
        torch.nn.init.constant_(self.conv.bias, 0.01)
        torch.nn.init.constant_(self.branch.weight, 0.02)
        torch.nn.init.constant_(self.branch.bias, 0.01)
        torch.nn.init.constant_(self.skip.weight, -0.03)
        torch.nn.init.constant_(self.skip.bias, -0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute one feature scale.

        :param inputs:  NCHW input features.
        :returns:       Downsampled features.
        """
        # Concatenating a signed branch shares a non-qmin zero point with ReLU,
        # reproducing the original checkpoint's C3Ghost quantization boundary.
        features = self.conv(inputs)
        return torch.cat((torch.relu(self.branch(features)), self.skip(features)), dim=1)


def make_checkpoint() -> SimpleNamespace:
    """Create a small real PyTorch OBB graph, without downloaded model files.

    :returns:   Loader-compatible checkpoint with 15 classes and 8 DFL bins.
    """
    head = torch.nn.Module()
    head.f, head.nc, head.reg_max, head.ne, head.nl = [2, 3, 4], 15, 8, 1, 3
    head.end2end = False
    for name, channels in (("cv2", 32), ("cv3", 15), ("cv4", 1)):
        branches = torch.nn.ModuleList(torch.nn.Conv2d(4, channels, 1) for _ in range(3))
        for branch in branches:
            torch.nn.init.constant_(branch.weight, 0.02)
            torch.nn.init.constant_(branch.bias, -4.0 if name == "cv3" else 0.1)
            if name == "cv3":
                with torch.no_grad():
                    branch.bias[0] = 2.0
        setattr(head, name, branches)
    model = torch.nn.Module()
    model.model = torch.nn.ModuleList([ConvLayer(3)] + [ConvLayer(4) for _ in range(4)] + [head])
    model.names = {index: f"known_{index}" for index in range(13)}
    return SimpleNamespace(task="obb", model=model)


def fixture_args(directory: Path, image_size: int = 256) -> object:
    """Create native-PT arguments and self-contained representative input.

    :param directory:   Temporary test directory.
    :param image_size:  Requested deployment resolution.
    :returns:           Parsed unified export arguments.
    """
    weights = directory / "checkpoint.pt"
    weights.write_bytes(b"fixture checkpoint resolved by the mocked loader")
    images = directory / "images"
    images.mkdir()
    Image.fromarray(np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)).save(images / "01.bmp")
    return parse_args(["--model", str(weights), "--to", "pte", "--image-size", str(image_size),
                       "--io-layout", "nhwc", "--calibration-dir", str(images),
                       "--validation-dir", str(images), "--calibration-limit", "1",
                       "--validation-limit", "1", "--threads", "2",
                       "--vela-config", str(ROOT / "scripts/vela/default_vela.ini"),
                       "--output-dir", str(directory / "export")],
                      source_formats=(".onnx", ".pt"))


class NativeObbExportTests(unittest.TestCase):
    """Keep the direct PT path independent of ONNX graph conversion."""

    def test_native_pt_pte_end_to_end(self):
        """Capture and quantize a native OBB graph, then check its actual serialized PTE."""
        with TemporaryDirectory() as temporary:
            args = fixture_args(Path(temporary))
            with patch("ultralytics.YOLO", return_value=make_checkpoint()), \
                 patch("torch.onnx.export", side_effect=AssertionError("ONNX export called")), \
                 patch("onnx2torch.convert", side_effect=AssertionError("ONNX adapter called")):
                report = run_export(args)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["source_input_channels"], 3)
            self.assertEqual(report["contract"]["input"]["shape"], [1, 1, 256, 256])
            self.assertEqual(report["labels"][-2:], ["class_13", "class_14"])
            pte = report["pte"]["Dedicated_Sram"]
            self.assertEqual(pte["cpu_operator_count"], 0)
            self.assertGreater(pte["npu_delegate_count"], 0)
            self.assertEqual(pte["serialized_tensor_layouts"]["input"]["shape"], [1, 256, 256, 1])
            self.assertEqual(len(pte["io_quantization"]["outputs"]), 4)
            self.assertTrue(Path(pte["path"]).is_file())
            self.assertFalse(list(args.output_dir.rglob("*.onnx")))
            self.assertEqual(json.loads((args.output_dir / "export_report.json").read_text())
                             ["status"], "completed")

    def test_invalid_size_records_failure(self):
        """Reject the generic 320 default rather than producing an incompatible qrcode PTE."""
        with TemporaryDirectory() as temporary:
            args = fixture_args(Path(temporary), image_size=320)
            with patch("ultralytics.YOLO", return_value=make_checkpoint()):
                with self.assertRaisesRegex(ValueError, "--image-size 256"):
                    run_export(args)
            self.assertEqual(json.loads((args.output_dir / "export_report.json").read_text())
                             ["status"], "failed")
            self.assertFalse(list(args.output_dir.rglob("*.pte")))

    def test_native_float_outputs_keep_integer_input(self):
        """Compile four model-side dequantizers while rejecting network CPU fallback."""
        with TemporaryDirectory() as temporary:
            args = fixture_args(Path(temporary))
            args.output_type = "float32"
            with patch("ultralytics.YOLO", return_value=make_checkpoint()):
                report = run_export(args)
            pte = report["pte"]["Dedicated_Sram"]
            self.assertEqual(pte["serialized_tensor_layouts"]["input"]["dtype"], "int8")
            self.assertTrue(all(item["dtype"] == "float32"
                                for item in pte["serialized_tensor_layouts"]["outputs"]))
            self.assertEqual(pte["cpu_operators"], ["cortex_m::dequantize_per_tensor"])
            self.assertEqual(pte["execution"]["boundary_dequantize_count"], 4)
            self.assertEqual(pte["execution"]["network_cpu_operator_count"], 0)

    def test_detect_keeps_existing_exporter(self):
        """Leave ordinary Detect checkpoints on their existing generic PT route."""
        with TemporaryDirectory() as temporary:
            args = fixture_args(Path(temporary))
            with patch("ultralytics.YOLO", return_value=SimpleNamespace(task="detect")), \
                 patch("export_yolo_pte.run_export", return_value={"fixture": True}) as exporter:
                report = run_export(args)
            exporter.assert_called_once()
            self.assertEqual(report["pte"]["Dedicated_Sram"], {"fixture": True})


if __name__ == "__main__":
    unittest.main()
