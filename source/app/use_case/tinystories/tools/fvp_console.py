# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Send prompts to the TinyStories Raw TCP UART and display streamed responses."""

import argparse
import codecs
import re
import socket
import sys
import time


class StoryOutput:
    """Stream story fragments; replace the FVP summary time with host wall time."""

    def __init__(self, started: float):
        self.started = started
        self.pending = ''
        self.line_start = True
        self.summary = False

    def write(self, text: str):
        """Write immediately except while recognizing or collecting a DONE line.

        :param text:  Decoded UART fragment.
        """
        prefix = 'INFO - TINYSTORIES_DEMO DONE'
        output = []
        for character in text:
            if self.summary:
                self.pending += character
                if character == '\n':
                    elapsed = (time.monotonic() - self.started) * 1000
                    line = self.pending.rstrip('\r\n')
                    line = re.sub(r' total_ms=[\d.]+', '', line)
                    output.append(f'{line} total_ms={elapsed:.3f} time_source=host_wall\n')
                    self.pending = ''
                    self.summary = False
                    self.line_start = True
            elif self.line_start:
                self.pending += character
                if self.pending == prefix:
                    self.summary = True
                elif not prefix.startswith(self.pending):
                    output.append(self.pending)
                    self.pending = ''
                    self.line_start = character == '\n'
            else:
                output.append(character)
                self.line_start = character == '\n'
        sys.stdout.write(''.join(output))
        sys.stdout.flush()

    def finish(self):
        """Flush any held prefix and separate the next local input prompt."""
        sys.stdout.write(self.pending + '\n')
        sys.stdout.flush()
        self.pending = ''


def receive_response(connection: socket.socket, timeout: float, started: float):
    """Stream a request through the next READY, excluding stale startup READY.

    :param connection:  Connected Raw TCP UART socket.
    :param timeout:     Maximum response wait in seconds.
    :param started:     Host monotonic timestamp before sending the prompt.
    """
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    output = StoryOutput(started)
    tail = b''
    started = False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            data = connection.recv(65536)
        except socket.timeout:
            continue
        if not data:
            raise ConnectionError('FVP closed the UART connection')
        output.write(decoder.decode(data))
        tail = (tail + data)[-8192:]
        started |= b'TINYSTORIES_DEMO BUSY' in tail or b'TINYSTORIES_DEMO rejected:' in tail
        if b'DEMO stopped' in tail or b'HardFault' in tail:
            raise RuntimeError('Firmware stopped; see the UART output above')
        if (started and b'TINYSTORIES_DEMO READY' in tail
                and tail.rstrip().endswith(b'Prompt>')):
            output.write(decoder.decode(b'', final=True))
            output.finish()
            return
    raise TimeoutError('No completed response; inspect the FVP window and UART port')


def send_prompt(connection: socket.socket, prompt: str, timeout: float):
    """Validate and send a single prompt using LF, then stream its response.

    :param connection:  Connected Raw TCP UART socket.
    :param prompt:      UTF-8 story opening.
    :param timeout:     Maximum response wait in seconds.
    """
    encoded = prompt.encode('utf-8')
    if not encoded or len(encoded) > 256 or any(byte in encoded for byte in (0, 10, 13)):
        raise ValueError('Use one nonempty line, at most 256 UTF-8 bytes')
    started = time.monotonic()
    connection.sendall(encoded + b'\n')
    receive_response(connection, timeout, started)


def main():
    """Connect after firmware READY and provide a prompt entry console."""
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--prompt',
                        help='Send one prompt and exit; otherwise use interactive input')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.timeout <= 0:
        parser.error('Port must be 1..65535 and timeout must be positive')
    with socket.create_connection((args.host, args.port), timeout=10) as connection:
        connection.settimeout(1)
        print('Connected. FVP must have reached READY. Generation can take several minutes.')
        if args.prompt is not None:
            send_prompt(connection, args.prompt, args.timeout)
            return
        print('Enter a story opening below; /quit disconnects this console.')
        while True:
            try:
                prompt = input('PC prompt> ')
            except EOFError:
                return
            if prompt == '/quit':
                return
            try:
                send_prompt(connection, prompt, args.timeout)
            except ValueError as error:
                print(error)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError) as error:
        print(f'FVP console: {error}', file=sys.stderr)
        sys.exit(1)
