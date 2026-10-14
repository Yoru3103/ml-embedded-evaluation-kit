# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Check NHWC interfaces, firmware image preparation and multi-head quantization."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from qrcode_onnx2tf_backend import check_nhwc_contract, load_image_array
from qrcode_tflite_utils import lite_runner


class QrcodeNhwcTests(unittest.TestCase):
    """Protect layout semantics and the existing interpreter's output mapping."""

    def test_centered_crop(self):
        """Reject the old right-biased crop by checking a non-square RGB gradient."""
        pixels = np.arange(6 * 4 * 3, dtype=np.uint8).reshape(4, 6, 3)
        with patch("qrcode_onnx2tf_backend.Image.open", return_value=Image.fromarray(pixels)):
            actual = load_image_array(Path("unused.png"), 4, "mlek-crop")
        expected = pixels[:, 1:5].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(actual.flags.c_contiguous)

    def test_nhwc_contract(self):
        """Require NHWC heads without changing the non-image auxiliary tensor."""
        source = {"outputs": [{"shape": [1, 79, size, size]} for size in (32, 16, 8)]
                             + [{"shape": [1, 1, 1344]}]}
        contract = {"input_layout": "NHWC",
                    "outputs": [{"shape": [1, size, size, 79]} for size in (32, 16, 8)]
                               + [{"shape": [1, 1, 1344]}]}
        check_nhwc_contract(contract, source)
        contract["outputs"][1]["shape"] = [1, 79, 16, 16]
        with self.assertRaises(ValueError):
            check_nhwc_contract(contract, source)

    def test_ordered_outputs_and_quantization(self):
        """Use independent output scales and restore reordered NHWC heads to NCHW."""
        shapes = [[1, 79, size, size] for size in (4, 2, 1)] + [[1, 1, 21]]
        metadata = {"input": {"shape": [1, 3, 32, 32]},
                    "outputs": [{"shape": shape} for shape in shapes]}
        details = []
        for index, shape in enumerate(shapes):
            nhwc = [shape[k] for k in (0, 2, 3, 1)] if len(shape) == 4 else shape
            details.append({"index": index + 1, "name": str(index), "shape": np.array(nhwc),
                            "dtype": np.int8, "quantization": (0.1 * (index + 1), index)})
        input_details = {"index": 0, "name": "image", "shape": np.array([1, 32, 32, 3]),
                         "dtype": np.int8, "quantization": (1 / 255, -128)}
        with patch("ai_edge_litert.interpreter.Interpreter") as factory:
            interpreter = factory.return_value
            interpreter.get_input_details.return_value = [input_details]
            interpreter.get_output_details.return_value = list(reversed(details))
            values = {item["index"]: np.full(item["shape"], 10, np.int8) for item in details}
            interpreter.get_tensor.side_effect = values.__getitem__
            predict, contract = lite_runner(Path("unused.tflite"), metadata, True)
            actual = predict(np.zeros([1, 3, 32, 32], np.float32))
            self.assertEqual(contract["input_layout"], "NHWC")
            for index, output in enumerate(actual):
                self.assertEqual(list(output.shape), shapes[index])
                np.testing.assert_allclose(output, (10 - index) * 0.1 * (index + 1))
            self.assertEqual(list(interpreter.set_tensor.call_args.args[1].shape), [1, 32, 32, 3])
            np.testing.assert_array_equal(interpreter.set_tensor.call_args.args[1], -128)


if __name__ == "__main__":
    unittest.main()
