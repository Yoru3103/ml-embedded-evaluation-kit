# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual console against a local UART protocol fixture."""

import socket
import subprocess
import sys
import threading
import unittest
from pathlib import Path

CLIENT = Path(__file__).resolve().parents[1] / 'tools' / 'fvp_console.py'


def serve(listener: socket.socket, count: int, received: list[bytes], errors: list[Exception]):
    """Serve bounded requests with split UTF-8, stale READY and trailing spaces.

    :param listener:  Bound loopback listener.
    :param count:     Expected request count.
    :param received:  Received prompt lines.
    :param errors:    Server errors reported to the test thread.
    """
    try:
        with listener.accept()[0] as connection:
            connection.settimeout(10)
            connection.sendall(b'TINYSTORIES_DEMO READY\r\nPrompt> ')
            for _ in range(count):
                line = bytearray()
                while not line.endswith(b'\n'):
                    block = connection.recv(1)
                    if not block:
                        raise ConnectionError('Client closed before sending a prompt')
                    line.extend(block)
                received.append(bytes(line))
                connection.sendall(b'TINYSTORIES_DEMO BUSY\r\nStory:\r\n\xe6')
                connection.sendall(b'\x95\x85\xe4\xba\x8b.\r\nINFO - TINYSTORIES_DEMO DO')
                connection.sendall(b'NE success=1 total_ms=123456789.123 generated=2\r\n')
                connection.sendall(b'TINYSTORIES_DEMO READY\r\nPrompt>  ')
    except (OSError, RuntimeError) as error:
        errors.append(error)


class ConsoleTest(unittest.TestCase):
    """Verify one-shot and interactive transports without changing a running FVP."""

    def run_console(self, prompts: list[str], interactive: bool):
        """Run a console process and validate every transmitted prompt.

        :param prompts:      Prompt texts to send.
        :param interactive:  Whether to use stdin rather than --prompt.
        """
        received, errors = [], []
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            listener.settimeout(10)
            worker = threading.Thread(target=serve, args=(listener, len(prompts), received, errors))
            worker.start()
            command = [sys.executable, str(CLIENT), '--port', str(listener.getsockname()[1])]
            if not interactive:
                command += ['--prompt', prompts[0]]
            result = subprocess.run(command, input='\n'.join(prompts + ['/quit']) + '\n',
                                    encoding='utf-8', capture_output=True, timeout=15,
                                    check=False)
            worker.join(2)
            self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(received, [prompt.encode('utf-8') + b'\n' for prompt in prompts])
        self.assertEqual(result.stdout.count('故事.'), len(prompts))
        self.assertEqual(result.stdout.count('DONE success=1'), len(prompts))
        self.assertEqual(result.stdout.count('time_source=host_wall'), len(prompts))
        self.assertEqual(result.stdout.count('total_ms='), len(prompts))
        self.assertNotIn('123456789.123', result.stdout)

    def test_single_prompt(self):
        """Send a complete Unicode argument and stream the response."""
        self.run_console(['A little cat 遇见 a robot.'], False)

    def test_interactive_prompts(self):
        """Wait for READY between two interactive prompts, then quit."""
        self.run_console(['Once upon a time,', 'A little cat'], True)


if __name__ == '__main__':
    unittest.main()
