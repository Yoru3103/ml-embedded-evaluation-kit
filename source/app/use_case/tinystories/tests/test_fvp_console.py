# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual console against a local UART protocol fixture."""

import io
import runpy
import socket
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

CLIENT = Path(__file__).resolve().parents[1] / 'tools' / 'fvp_console.py'


def read_line(connection: socket.socket) -> bytes:
    """Read one LF-terminated line.

    :param connection:  Accepted client socket.
    :returns:           Bytes including the terminator.
    """
    line = bytearray()
    while not line.endswith(b'\n'):
        block = connection.recv(1)
        if not block:
            raise ConnectionError('Client closed before sending a line')
        line.extend(block)
    return bytes(line)


def serve(listener: socket.socket, count: int, received: list[bytes], errors: list[Exception],
          greedy: bool = False):
    """Serve bounded requests with split UTF-8, stale READY and trailing spaces.

    :param listener:  Bound loopback listener.
    :param count:     Expected request count.
    :param received:  Received command and prompt lines.
    :param errors:    Server errors reported to the test thread.
    :param greedy:    Expect a leading ``!greedy`` command.
    """
    try:
        with listener.accept()[0] as connection:
            connection.settimeout(10)
            connection.sendall(b'TINYSTORIES_DEMO READY\r\nPrompt> ')
            if greedy:
                received.append(read_line(connection))
                connection.sendall(b'INFO - TINYSTORIES_DEMO decode=greedy\r\n'
                                   b'TINYSTORIES_DEMO READY\r\nPrompt> ')
            for _ in range(count):
                received.append(read_line(connection))
                connection.sendall(b'TINYSTORIES_DEMO BUSY\r\nStory:\r\n\xe6')
                connection.sendall(b'\x95\x85\xe4\xba\x8b.\r\nINFO - TINYSTORIES_DEMO DO')
                connection.sendall(b'NE success=1 total_ms=123456789.123 generated=2\r\n')
                connection.sendall(b'TINYSTORIES_DEMO READY\r\nPrompt>  ')
    except (OSError, RuntimeError) as error:
        errors.append(error)


class ConsoleTest(unittest.TestCase):
    """Verify one-shot and interactive transports without changing a running FVP."""

    def test_sentence_streams_before_newline(self):
        """Display an unterminated UART sentence as soon as its punctuation arrives."""
        output = io.StringIO()
        story = runpy.run_path(str(CLIENT))['StoryOutput']()
        with patch('sys.stdout', output):
            story.write('Story:\r\nA little cat')
            self.assertEqual(output.getvalue(), '')
            story.write('.')
            self.assertEqual(output.getvalue(), 'A little cat.')
            story.write(' The rest\r\nINFO - TINYSTORIES_DEMO DONE success=1\r\n')
            self.assertEqual(output.getvalue(), 'A little cat. The rest\n')

    def run_console(self, prompts: list[str], interactive: bool, greedy: bool = False):
        """Run a console process and validate every transmitted prompt.

        :param prompts:      Prompt texts to send.
        :param interactive:  Whether to use stdin rather than --prompt.
        :param greedy:       Whether to pass ``--greedy``.
        """
        received, errors = [], []
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            listener.settimeout(10)
            worker = threading.Thread(target=serve,
                                      args=(listener, len(prompts), received, errors, greedy))
            worker.start()
            command = [sys.executable, str(CLIENT), '--port', str(listener.getsockname()[1])]
            if greedy:
                command += ['--greedy']
            if not interactive:
                command += ['--prompt', prompts[0]]
            result = subprocess.run(command, input='\n'.join(prompts + ['/quit']) + '\n',
                                    encoding='utf-8', capture_output=True, timeout=15,
                                    check=False)
            worker.join(2)
            self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = [b'!greedy\n'] if greedy else []
        expected += [prompt.encode('utf-8') + b'\n' for prompt in prompts]
        self.assertEqual(received, expected)
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

    def test_greedy_command(self):
        """Send !greedy and wait for its READY before the story prompt."""
        self.run_console(['Once upon a time,'], False, True)


if __name__ == '__main__':
    unittest.main()
