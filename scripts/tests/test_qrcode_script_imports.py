# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Check Python snippets embedded in the Linux matrix entry points."""

from importlib import import_module
from pathlib import Path
import re
import unittest


class MatrixScriptImportsTests(unittest.TestCase):
    """Catch module moves that leave otherwise valid Bash scripts unusable."""

    def test_embedded_python_imports(self):
        """Resolve both matrix scripts' export helpers and their public symbols."""
        root = Path(__file__).resolve().parents[1] / "qrcode"
        imports = 0
        for name in ("build_qrcode_pte_matrix.sh", "build_qrcode_tflite_matrix.sh"):
            text = (root / name).read_text(encoding="utf-8")
            for module, symbols in re.findall(r"^from (\S+) import ([^\n]+)", text, re.MULTILINE):
                if not module.startswith(("model_export.", "qrcode_")):
                    continue
                with self.subTest(script=name, module=module):
                    self.assertTrue(module.startswith("model_export."))
                    loaded = import_module(module)
                    for symbol in symbols.strip().split(", "):
                        self.assertTrue(hasattr(loaded, symbol))
                imports += 1
        self.assertGreaterEqual(imports, 4)


if __name__ == "__main__":
    unittest.main()
