# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its
# affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for shell-free Windows/POSIX process arguments."""
import json
import subprocess
import sys
import unittest

from mlek_tools.setup.util import call_command


class CommandArgumentsTests(unittest.TestCase):
    """Check that host paths and shell metacharacters survive unchanged."""

    def test_argument_round_trip(self):
        """Preserve spaces, backslashes, quotes, semicolons and shell characters."""
        expected = [
            r"D:\Program Files\Arm\bin",
            r"C:\models\customer model.tflite",
            "literal & value", "one;two", "a'b", 'a"b', "%PATH%",
        ]
        result = call_command([
            sys.executable, "-c",
            "import json, sys; print(json.dumps(sys.argv[1:]))",
            *expected,
        ])
        self.assertEqual(json.loads(result), expected)

    def test_nonzero_exit(self):
        """Propagate a child process failure to the caller."""
        with self.assertRaises(subprocess.CalledProcessError) as error:
            call_command([sys.executable, "-c", "raise SystemExit(7)"])
        self.assertEqual(error.exception.returncode, 7)


if __name__ == "__main__":
    unittest.main()
