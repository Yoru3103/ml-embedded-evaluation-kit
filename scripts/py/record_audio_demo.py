# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Record one complete utterance without energy gating or spectral processing."""
import importlib
from typing import Any


def record_utterance(audio: Any, device: Any, seconds: float, cancel: Any) -> bytes:
    """Show Start/Stop buttons and return captured, unmodified PCM16.

    :param audio:   Sounddevice module.
    :param device:  Optional input device index.
    :param seconds: Maximum recording duration.
    :param cancel:  Host shutdown event.
    :returns:       Mono 16 kHz PCM16 bytes, at most the requested duration.
    """
    tk = importlib.import_module("tkinter")
    root = tk.Tk()
    root.title("Chinese ASR | record one sentence")
    root.geometry("420x150")
    chunks = []
    stream = None
    status = tk.StringVar(value="Start recording, then Stop to recognize (max 30s).")

    def capture(data: Any, _frames: int, _time: Any, flags: Any):
        """Keep raw blocks; reject hardware overflow instead of hiding lost audio."""
        if flags:
            cancel.set()
        chunks.append(bytes(data))

    def finish():
        """End capture and return to the existing VSI producer."""
        root.quit()

    def start():
        """Open the microphone only after an explicit button click."""
        nonlocal stream
        stream = audio.RawInputStream(samplerate=16000, blocksize=320, channels=1,
                                      dtype="int16", device=device, callback=capture)
        stream.start()
        begin.config(state="disabled")
        status.set("Recording... Stop to recognize.")
        root.after(int(seconds * 1000), finish)

    def poll():
        """Close the capture window when the owning demo stops."""
        if cancel.is_set():
            finish()
        else:
            root.after(100, poll)

    tk.Label(root, textvariable=status).pack(pady=15)
    begin = tk.Button(root, text="Start recording", command=start)
    begin.pack(side="left", padx=50)
    tk.Button(root, text="Stop / Recognize", command=finish).pack(side="right", padx=40)
    root.protocol("WM_DELETE_WINDOW", finish)
    root.after(100, poll)
    try:
        root.mainloop()
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        root.destroy()
    if cancel.is_set():
        raise RuntimeError("Recording cancelled or microphone overflowed")
    return b"".join(chunks)[:int(seconds * 16000) * 2]
