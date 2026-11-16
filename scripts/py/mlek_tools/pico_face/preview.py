# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Display and validate the eight-step pico-face FVP preview."""

import argparse
from pathlib import Path
import re
import subprocess
import tempfile
import time
import zlib

FRAME = re.compile(rb"PICO_FACE_FRAME step=(\d+) crc=([0-9a-f]{8})\r?\n")
LAST_STEP = 8
FRAME_BYTES = 128 * 128 * 3
PREVIEW_SCALE = 2
PREVIEW_SIZE = 128 * PREVIEW_SCALE


class PreviewRun:
    """Own one FVP process, its log and the validated frames."""

    def __init__(self, directory: Path, command: list[str], timeout: int, seed: int, class_id: int):
        """Start the simulator in an isolated output directory."""
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="run-", dir=directory))
        # The firmware reads a little-endian uint32 seed followed by one class byte.
        request = seed.to_bytes(4, "little") + bytes([class_id])
        (self.directory / "pico_face_preview.bin").write_bytes(request)
        self.log = self.directory / "fvp.log"
        self.frames: list[bytes] = []
        self.deadline = time.monotonic() + timeout
        with self.log.open("wb") as output:
            self.process = subprocess.Popen(
                command, cwd=self.directory, stdout=output, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        print(f"FVP output: {self.directory}", flush=True)

    def poll(self) -> bool:
        """Validate newly published frames and report successful completion."""
        output = self.log.read_bytes()
        for match in list(FRAME.finditer(output))[len(self.frames):]:
            step, crc = int(match[1]), int(match[2], 16)
            if step != len(self.frames) or step > LAST_STEP:
                raise ValueError(f"Unexpected frame step {step}")
            name = "pico_face.rgb" if step == LAST_STEP else f"pico_face_{step}.rgb"
            frame = (self.directory / name).read_bytes()
            if len(frame) != FRAME_BYTES or zlib.crc32(frame) != crc:
                raise ValueError(f"Invalid RGB length or CRC for step {step}")
            self.frames.append(frame)

        result = self.process.poll()
        if result is None:
            if time.monotonic() >= self.deadline:
                raise TimeoutError(f"FVP timed out. See {self.log}")
            return False
        # The process can exit after the first log read. Consume its final output.
        if self.log.stat().st_size != len(output):
            return self.poll()
        if result or len(self.frames) != LAST_STEP + 1 or b"PICO_FACE_PASS" not in output:
            raise RuntimeError(f"FVP did not complete the preview (exit {result}). See {self.log}")
        return True

    def close(self):
        """Stop the simulator when the window closes or validation fails."""
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()


def show_preview(directory: Path, command: list[str], timeout: int, seed: int, class_id: int) -> int:
    """Open a window for starting a run, viewing frames and replaying them."""
    import tkinter as tk  # Imported only for the interactive mode.

    root = tk.Tk()
    root.title("Pico-face · Corstone-320 FVP")
    status = tk.StringVar(value=f"Seed {seed} · Class {class_id} · Ready to generate")
    blank = tk.PhotoImage(width=PREVIEW_SIZE, height=PREVIEW_SIZE)
    picture = tk.Label(root, image=blank, bg="#171a20")
    picture.pack(padx=16, pady=16)
    tk.Label(root, textvariable=status, wraplength=PREVIEW_SIZE).pack()
    controls = tk.Frame(root)
    controls.pack(pady=12)
    run = None
    failed = False
    shown = 0

    def show(step: int):
        """Render an RGB frame using Tk's PPM support."""
        data = b"P6\n128 128\n255\n" + run.frames[step]
        picture.image = tk.PhotoImage(data=data, format="PPM").zoom(PREVIEW_SCALE)
        picture.configure(image=picture.image)
        status.set(f"Seed {seed} · Class {class_id} · Step {step}/{LAST_STEP}")

    def check():
        """Poll for new frames without blocking the window."""
        nonlocal failed, shown
        try:
            complete = run.poll()
            if shown < len(run.frames):
                show(shown)
                shown += 1
                root.after(120, check)
                return
            if not complete:
                root.after(50, check)
                return
            replay.configure(state=tk.NORMAL)
        except (OSError, RuntimeError, ValueError) as error:
            failed = True
            run.close()
            status.set(str(error))
        generate.configure(state=tk.NORMAL)

    def start():
        """Start a new run after the preceding run or replay has finished."""
        nonlocal run, failed, shown
        failed = False
        shown = 0
        try:
            run = PreviewRun(directory, command, timeout, seed, class_id)
        except OSError as error:
            failed = True
            status.set(str(error))
            return
        generate.configure(state=tk.DISABLED)
        replay.configure(state=tk.DISABLED)
        picture.configure(image=blank)
        status.set("Starting FVP…")
        check()

    def play(step: int = 0):
        """Replay the validated sequence at a fixed interval."""
        generate.configure(state=tk.DISABLED)
        replay.configure(state=tk.DISABLED)
        show(step)
        if step < LAST_STEP:
            root.after(350, play, step + 1)
        else:
            generate.configure(state=tk.NORMAL)
            replay.configure(state=tk.NORMAL)

    generate = tk.Button(controls, text="Generate", command=start)
    generate.pack(side=tk.LEFT, padx=4)
    replay = tk.Button(controls, text="Replay", command=play, state=tk.DISABLED)
    replay.pack(side=tk.LEFT, padx=4)
    try:
        root.mainloop()
    finally:
        if run is not None:
            run.close()
    return int(failed)


def main() -> int:
    """Run the interactive preview or validate one headless generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--class-id", type=int, choices=range(5), default=1)
    parser.add_argument("directory", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 <= args.seed <= 0xFFFFFFFF:
        parser.error("Seed must be between 0 and 4294967295")
    if not args.command:
        parser.error("An FVP command is required")
    if not args.headless:
        return show_preview(args.directory, args.command, args.timeout, args.seed, args.class_id)

    run = PreviewRun(args.directory, args.command, args.timeout, args.seed, args.class_id)
    try:
        while not run.poll():
            time.sleep(0.05)
        print(f"Validated {len(run.frames)} frames. Final CRC32: {zlib.crc32(run.frames[-1]):08x}")
    finally:
        run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
