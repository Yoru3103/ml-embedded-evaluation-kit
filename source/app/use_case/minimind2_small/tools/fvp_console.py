# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Send single-turn MiniMind chat requests over the FVP raw UART socket."""

import argparse
import codecs
import socket
import sys
from pathlib import Path
from typing import BinaryIO

from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
from tokenizers import Tokenizer

def encode_prompt(
    tokenizer: Tokenizer, template: str, text: str, context_length: int
) -> tuple[list[int], bool]:
    """Render the chat template and trim the oldest user tokens until it fits."""
    prompt_limit = context_length - 1
    if prompt_limit < 1:
        raise ValueError("Context length must leave room for at least one prompt token")
    user_tokens = tokenizer.encode(text, add_special_tokens=False).ids
    chat = SandboxedEnvironment(undefined=StrictUndefined).from_string(template)
    keep = min(len(user_tokens), prompt_limit)
    while keep >= 0:
        content = text if keep == len(user_tokens) else tokenizer.decode(
            user_tokens[len(user_tokens) - keep:], skip_special_tokens=False
        )
        rendered = chat.render(
            messages=[{"role": "user", "content": content}], tools=None,
            add_generation_prompt=True,
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        if len(ids) <= prompt_limit:
            return ids, keep < len(user_tokens)
        keep -= 1
    raise ValueError("Chat template alone exceeds the prompt budget")


def read_line(stream: BinaryIO) -> bytes:
    """Read one UART line and report an unexpected disconnect."""
    line = stream.readline()
    if not line:
        raise EOFError("FVP UART closed")
    return line.rstrip(b"\r\n")


def read_exact(stream: BinaryIO, length: int) -> bytes:
    """Read an exact binary payload from a framed UART stream."""
    payload = bytearray()
    while len(payload) < length:
        block = stream.read(length - len(payload))
        if not block:
            raise EOFError("Incomplete MiniMind token payload")
        payload.extend(block)
    return bytes(payload)


def read_ready(stream: BinaryIO) -> int:
    """Wait for a READY frame and return the firmware's static context size."""
    while True:
        line = read_line(stream)
        if line.startswith(b"MINIMIND_DEMO READY "):
            fields = dict(part.split(b"=", 1) for part in line.split()[2:])
            return int(fields[b"context"])


def main():
    """Connect to FVP and handle one terminal prompt at a time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--encode-stdin", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15650)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--context-length", type=int, choices=(64, 128), default=64)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    template = (args.tokenizer / "chat_template.jinja").read_text(encoding="utf-8")
    if args.encode_stdin:
        ids, truncated = encode_prompt(
            tokenizer,
            template,
            sys.stdin.buffer.read().decode("utf-8").rstrip("\r\n"),
            args.context_length,
        )
        if truncated:
            print(f"Prompt truncated to {len(ids)} tokens", file=sys.stderr)
        print(" ".join(map(str, ids)))
        return
    print(f"连接 FVP UART {args.host}:{args.port}，等待 READY…", flush=True)
    with socket.create_connection((args.host, args.port), timeout=10) as connection:
        connection.settimeout(15)
        # Request a fresh READY even if FVP emitted its initial banner before connection.
        connection.sendall(b"6400\n")
        with connection.makefile("rb") as stream:
            try:
                while read_line(stream) != b"MINIMIND_DEMO ERROR invalid_prompt":
                    pass
                context_length = read_ready(stream)
            except (TimeoutError, EOFError) as error:
                raise SystemExit(
                    "UART 握手失败。请核对 FVP 的 telnetterminal0 端口并用 --port 指定。"
                ) from error
            connection.settimeout(args.timeout)
            while True:
                try:
                    text = input("你 > ")
                except (EOFError, KeyboardInterrupt):
                    return
                while not text.strip():
                    try:
                        text = input("你 > ")
                    except (EOFError, KeyboardInterrupt):
                        return
                ids, truncated = encode_prompt(tokenizer, template, text, context_length)
                if truncated:
                    print(f"[输入已截断，模板和用户文本共 {len(ids)} tokens]")
                connection.sendall((" ".join(map(str, ids)) + "\n").encode("ascii"))
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                while True:
                    line = read_line(stream)
                    if line.startswith(b"MINIMIND_DEMO ERROR "):
                        print(line.decode("ascii"))
                        break
                    if line.startswith(b"MINIMIND_DEMO START "):
                        print("MiniMind > ", end="", flush=True)
                        continue
                    if line.startswith(b"MINIMIND_DEMO CHUNK "):
                        _, _, _, byte_count = line.split()
                        payload = read_exact(stream, int(byte_count))
                        sys.stdout.write(decoder.decode(payload, final=False))
                        sys.stdout.flush()
                        continue
                    if line.startswith(b"MINIMIND_DEMO DONE "):
                        sys.stdout.write(decoder.decode(b"", final=True))
                        sys.stdout.write("\n")
                        fields = dict(part.split(b"=", 1) for part in line.split()[2:])
                        print(f"[{fields[b'reason'].decode('ascii')}]")
                        break
                read_ready(stream)


if __name__ == "__main__":
    main()
