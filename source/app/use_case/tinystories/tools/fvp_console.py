# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Send prompts to the TinyStories Raw TCP UART and display streamed story text."""

import argparse
import codecs
import re
import socket
import sys
import time

_MARKERS = ('INFO - ', 'WARN - ', 'ERROR - ', 'E [', 'I [', 'TINYSTORIES_', 'Prompt> ')
_SKIP_FRAGMENTS = ('Processor internal clock', 'Application Note', 'MPS4 build', 'MPS4 core clock', 'V2M-MPS4')
_READY = b'TINYSTORIES_DEMO READY'
_DONE = b'TINYSTORIES_DEMO DONE '
_REJECTED = b'TINYSTORIES_DEMO rejected:'


def clean_line(line: str) -> str:
    """Remove firmware log text from one UART line.

    :param line:  UART line without newline.
    :returns:     Story text, or an empty string for a log-only line.
    """
    for marker in _MARKERS:
        index = line.find(marker)
        if index >= 0:
            line = line[:index]
    line = line.replace('Story:', '').strip('\r')
    for fragment in _SKIP_FRAGMENTS:
        if fragment in line:
            return ''
    return line


class StoryOutput:
    """Buffer complete UART lines and print only their story prefix."""

    def __init__(self, prompt: str = ''):
        self.buffer = ''
        self.skip_prompt = prompt.strip()
        self.in_story = False

    def write(self, text: str):
        """Consume decoded UART text and print story lines.

        :param text:  Decoded UART fragment.
        """
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.rstrip('\r') == 'Story:':
                self.in_story = True
                self.skip_prompt = ''
                continue
            cleaned = clean_line(line)
            if self.skip_prompt and cleaned.strip() == self.skip_prompt:
                self.skip_prompt = ''
                continue
            if cleaned.strip() or (self.in_story and not line.strip()):
                sys.stdout.write(cleaned + '\n')
                sys.stdout.flush()

        # Firmware writes sentence fragments without adding a newline.
        if self.in_story and not any(self.buffer.startswith(marker) for marker in _MARKERS):
            boundary = max(self.buffer.rfind(mark) for mark in '.!?') + 1
            if boundary:
                sys.stdout.write(self.buffer[:boundary])
                sys.stdout.flush()
                self.buffer = self.buffer[boundary:]

    def finish(self):
        """Flush the last partial line."""
        cleaned = clean_line(self.buffer)
        self.buffer = ''
        if cleaned.strip():
            sys.stdout.write(cleaned + '\n')
            sys.stdout.flush()


def receive_response(connection: socket.socket, timeout: float, started: float, prompt: str = ''):
    """Stream a request through the next READY, excluding stale startup READY.

    :param connection:  Connected Raw TCP UART socket.
    :param timeout:     Maximum response wait in seconds.
    :param started:     Host monotonic timestamp before sending the prompt.
    :param prompt:      Prompt to suppress if echoed by firmware.
    """
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    output = StoryOutput(prompt)
    tail = b''
    finished = None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            data = connection.recv(65536)
        except socket.timeout:
            continue
        if not data:
            raise ConnectionError('FVP closed the UART connection')
        received = time.monotonic()
        output.write(decoder.decode(data))
        tail = (tail + data)[-8192:]
        if b'DEMO stopped' in tail or b'HardFault' in tail:
            raise RuntimeError('Firmware stopped; see the UART output above')
        ready = tail.rfind(_READY)
        done = tail.rfind(_DONE)
        rejected = tail.rfind(_REJECTED)
        if finished is None and done >= 0 and b'\n' in tail[done:]:
            finished = received
        if 0 <= max(done, rejected) < ready and tail.rstrip().endswith(b'Prompt>'):
            output.write(decoder.decode(b'', final=True))
            output.finish()
            if rejected > done:
                raise ValueError(tail[rejected:].splitlines()[0].decode('utf-8', 'replace'))
            summary = tail[done:].splitlines()[0].decode('utf-8', 'replace')
            elapsed_ms = ((finished if finished is not None else received) - started) * 1000
            summary = re.sub(r'\btotal_ms=\S+', f'total_ms={elapsed_ms:.3f}', summary)
            print(f'{summary} time_source=host_wall', flush=True)
            return
    raise TimeoutError('No completed response; inspect the FVP window and UART port')


def wait_for_greedy(connection: socket.socket, timeout: float):
    """Read until the firmware acknowledges greedy mode and returns to READY.

    :param connection:  Connected Raw TCP UART socket.
    :param timeout:     Maximum wait in seconds.
    """
    deadline = time.monotonic() + timeout
    tail = b''
    while time.monotonic() < deadline:
        try:
            data = connection.recv(65536)
        except socket.timeout:
            continue
        if not data:
            raise ConnectionError('FVP closed the UART connection')
        tail = (tail + data)[-8192:]
        ack = tail.rfind(b'TINYSTORIES_DEMO decode=greedy')
        if 0 <= ack < tail.rfind(_READY) and tail.rstrip().endswith(b'Prompt>'):
            return
    raise TimeoutError('No READY from the firmware')


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
    receive_response(connection, timeout, started, prompt)


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
    parser.add_argument('--greedy', action='store_true',
                        help='Send !greedy before the first prompt for deterministic output')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.timeout <= 0:
        parser.error('Port must be 1..65535 and timeout must be positive')
    with socket.create_connection((args.host, args.port), timeout=10) as connection:
        connection.settimeout(1)
        if args.greedy:
            connection.sendall(b'!greedy\n')
            wait_for_greedy(connection, args.timeout)
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
    except (OSError, RuntimeError, ValueError, TimeoutError, ConnectionError) as error:
        print(f'FVP console: {error}', file=sys.stderr)
        sys.exit(1)
