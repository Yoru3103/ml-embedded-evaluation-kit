# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Check source-based routing before any expensive model conversion starts."""

from pathlib import Path
import json
import os
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ai_edge_litert import schema_py_generated as schema
import flatbuffers
import numpy as np
import onnx
from PIL import Image

from export_model import run_export, select_tflite_backend
from export_qrcode_onnx import parse_args, run_exports
from test_qrcode_contract import make_obb_graph
from qrcode_tflite_utils import validate_tflite_models
import qrcode_onnx2tf


def make_tflite(integer: bool) -> bytes:
    """Build constant outputs whose values match the small ONNX fixture.

    :param integer: Use INT8 tensors with explicit quantization parameters.
    :returns:       Valid TFLite bytes with NHWC images and unchanged angle output.
    """
    model = schema.ModelT()
    model.version = 3
    graph = schema.SubGraphT()
    graph.tensors = []
    model.buffers = [schema.BufferT()]
    shapes = [[1, 32, 32, 3], [1, 4, 4, 65], [1, 2, 2, 65], [1, 1, 1, 65], [1, 1, 21]]
    for index, shape in enumerate(shapes):
        tensor = schema.TensorT()
        tensor.name, tensor.shape = f"tensor{index}", shape
        tensor.type = schema.TensorType.INT8 if integer else schema.TensorType.FLOAT32
        if integer:
            tensor.quantization = schema.QuantizationParametersT()
            tensor.quantization.scale, tensor.quantization.zeroPoint = [0.25], [0]
        if index:
            buffer = schema.BufferT()
            buffer.data = np.zeros(np.prod(shape), dtype=np.int8 if integer else np.float32)
            buffer.data = np.frombuffer(buffer.data.tobytes(), dtype=np.uint8)
            model.buffers.append(buffer)
            tensor.buffer = index
        graph.tensors.append(tensor)
    graph.inputs, graph.outputs = [0], [1, 2, 3, 4]
    model.subgraphs = [graph]
    builder = flatbuffers.Builder(256)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def convert_fixture(**options: dict):
    """Stand in for the external converter while retaining real interface validation.

    :param options: ONNX2TF conversion arguments.
    """
    calibration = np.load(options["custom_input_op_name_np_data_path"][0][1])
    np.testing.assert_array_equal(calibration, np.full((1, 32, 32, 3), 1.0, np.float32))
    staged = onnx.load(options["input_onnx_file_path"])
    shapes = {value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
              for value in staged.graph.value_info}
    np.testing.assert_array_equal(shapes["box_8"], [1, 64, 4, 4])
    np.testing.assert_array_equal(shapes["class_8"], [1, 1, 4, 4])
    directory = Path(options["output_folder_path"])
    for suffix, integer in (("float32", False), ("full_integer_quant", True)):
        (directory / f"network_{suffix}.tflite").write_bytes(make_tflite(integer))
    Path(options["input_onnx_file_path"]).write_bytes(b"converter rewrote its input")


class ExportModelTests(unittest.TestCase):
    """Keep ONNX conversion direct while retaining the native PyTorch path."""

    def test_automatic_backend(self):
        """Choose a converter that accepts the original model format."""
        self.assertEqual(select_tflite_backend(Path("network.onnx"), "auto"), "onnx2tf")
        self.assertEqual(select_tflite_backend(Path("network.pt"), "auto"), "litert")

    def test_explicit_backend(self):
        """Retain the alternative ONNX-to-PyTorch-to-LiteRT path."""
        self.assertEqual(select_tflite_backend(Path("network.onnx"), "litert"), "litert")
        with self.assertRaises(ValueError):
            select_tflite_backend(Path("network.pt"), "onnx2tf")

    def test_unknown_source(self):
        """Reject unsupported sources instead of guessing a conversion route."""
        with self.assertRaises(ValueError):
            select_tflite_backend(Path("network.bin"), "auto")

    def test_isolated_converter_api(self):
        """Forward converter options safely to an independent Python environment."""
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "onnx2tf.py").write_text(
                "import json\nfrom pathlib import Path\n__version__ = 'fixture'\n"
                "def convert(**options):\n"
                "    Path(options['output_folder_path'], 'options.json').write_text("
                "json.dumps(options))\n")
            options = {"input_onnx_file_path": "model with spaces.onnx",
                       "output_folder_path": str(directory),
                       "output_integer_quantized_tflite": True}
            with patch("qrcode_onnx2tf.CONVERTER_PYTHON", Path(sys.executable)), \
                 patch.dict(os.environ, {"PYTHONPATH": str(directory)}):
                environment = qrcode_onnx2tf.convert_onnx(options)
            self.assertEqual(json.loads((directory / "options.json").read_text()), options)
            self.assertEqual(environment["onnx2tf"], "fixture")

    def test_direct_onnx_tflite(self):
        """Produce validated integer artifacts even when PyTorch adaptation is unavailable."""
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = directory / "network.onnx"
            onnx.save(make_obb_graph(3, 16, 1), model)
            original = model.read_bytes()
            calibration = directory / "images"
            calibration.mkdir()
            Image.new("RGB", (32, 32), "white").save(calibration / "sample.png")
            config = directory / "vela.ini"
            config.write_text("")
            args = parse_args(["--model", str(model), "--to", "tflite",
                               "--calibration-dir", str(calibration), "--vela-config", str(config),
                               "--output-dir", str(directory / "export")])
            args.output_dir.mkdir()
            report = {}
            with patch("qrcode_onnx_utils.adapt_model", side_effect=AssertionError("PT used")), \
                 patch("qrcode_onnx2tf.CONVERTER_PYTHON", directory / "missing-python"), \
                 patch("qrcode_onnx2tf.import_module",
                       return_value=SimpleNamespace(convert=convert_fixture)), \
                 patch("qrcode_tflite_utils.compile_tflite", return_value={}):
                run_exports(args, report)
            self.assertEqual(report["tflite"]["backend"], "onnx2tf")
            self.assertEqual(report["tflite"]["int8"]["tensor_contract"]["input_layout"], "NHWC")
            self.assertTrue(Path(report["tflite"]["int8"]["path"]).is_file())
            self.assertNotIn("float_validation", report)
            self.assertEqual(model.read_bytes(), original)

    def test_converter_exit_records_failure(self):
        """Persist a failed report when the third-party converter exits the process."""
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = directory / "network.onnx"
            onnx.save(make_obb_graph(3, 16, 1), model)
            Image.new("RGB", (32, 32), "white").save(directory / "sample.png")
            config = directory / "vela.ini"
            config.write_text("")
            args = parse_args(["--model", str(model), "--to", "tflite",
                               "--calibration-dir", str(directory), "--vela-config", str(config),
                               "--output-dir", str(directory / "export")])
            converter = SimpleNamespace(convert=Mock(side_effect=SystemExit(1)))
            with patch("qrcode_onnx2tf.import_module", return_value=converter), \
                 patch("qrcode_onnx2tf.CONVERTER_PYTHON", directory / "missing-python"):
                with self.assertRaises(RuntimeError):
                    run_export(args)
            report = json.loads((args.output_dir / "export_report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertNotIn("tflm", report)

    def test_qoperator_rejects_recalibration(self):
        """Keep existing board calibration intact when native conversion is unsupported."""
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = directory / "boardcal.onnx"
            graph = make_obb_graph(3, 16, 1)
            graph.graph.node[0].op_type = "QLinearConv"
            onnx.save(graph, model)
            original = model.read_bytes()
            args = SimpleNamespace(output_dir=directory / "export")
            with patch("qrcode_onnx2tf.convert_onnx") as converter:
                with self.assertRaisesRegex(ValueError, "boardcal"):
                    qrcode_onnx2tf.export_tflite(model, [], [], {}, args)
            converter.assert_not_called()
            self.assertEqual(model.read_bytes(), original)

    def test_failed_validation_retains_diagnostics(self):
        """Keep converter models for inspection without replacing validated artifacts."""
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = directory / "network.onnx"
            onnx.save(make_obb_graph(3, 16, 1), model)
            Image.new("RGB", (32, 32), "white").save(directory / "sample.png")
            config = directory / "vela.ini"
            config.write_text("")
            args = parse_args(["--model", str(model), "--to", "tflite",
                               "--calibration-dir", str(directory), "--vela-config", str(config),
                               "--output-dir", str(directory / "export")])
            previous = args.output_dir / "tflite/best_int8.tflite"
            previous.parent.mkdir(parents=True)
            previous.write_bytes(b"last validated model")
            with patch("qrcode_onnx2tf.CONVERTER_PYTHON", directory / "missing-python"), \
                 patch("qrcode_onnx2tf.import_module",
                       return_value=SimpleNamespace(convert=convert_fixture)), \
                 patch("qrcode_onnx2tf.validate_tflite_models",
                       side_effect=ValueError("quantization limit")), \
                 patch("qrcode_tflite_utils.compile_tflite") as compile_model:
                with self.assertRaisesRegex(ValueError, "quantization limit"):
                    run_export(args)
            compile_model.assert_not_called()
            self.assertEqual(previous.read_bytes(), b"last validated model")
            self.assertTrue((previous.parent / "onnx2tf_outputs/network_float32.tflite").is_file())
            self.assertTrue((previous.parent /
                             "onnx2tf_outputs/network_full_integer_quant.tflite").is_file())
            report = json.loads((args.output_dir / "export_report.json").read_text())
            self.assertEqual(report["status"], "failed")

    def test_reject_wrong_tflite_layout(self):
        """Reject a converter that silently emits a different deployment interface."""
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.tflite"
            path.write_bytes(make_tflite(False))
            metadata = {"input": {"shape": [1, 3, 32, 32]},
                        "outputs": [{"shape": shape} for shape in
                                    [[1, 65, 4, 4], [1, 65, 2, 2],
                                     [1, 65, 1, 1], [1, 1, 21]]]}
            args = SimpleNamespace(io_layout="nchw", max_score_error=0.15, min_box_iou=0.9)
            with self.assertRaisesRegex(ValueError, "layout"):
                validate_tflite_models(path, path, [], metadata, args)


if __name__ == "__main__":
    unittest.main()
