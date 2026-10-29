# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Compare actual Cortex-M/U85 FVP scores with the exported PT2E reference."""
import argparse
import importlib
import json
from pathlib import Path
import re
import wave

import numpy as np
import torch
import torchaudio


def read_scores(log: Path) -> dict:
    """Parse one complete score matrix per logged inference window.

    :param log: FVP UART log from a DUMP_OUTPUT build.
    :returns:   Matrices keyed by clip index and window start sample.
    """
    text = log.read_text(errors="replace")
    if "Main loop terminated successfully" not in text:
        raise ValueError("Firmware did not complete successfully")
    pattern = re.compile(r"KWS_ZH score clip=(\d+) start=(\d+) frame=(\d+) "
                         r"value0=([\d.e+-]+) value1=([\d.e+-]+)")
    result = {}
    for clip, start, frame, first, second in pattern.findall(text):
        key = (int(clip), int(start))
        if key not in result:
            result[key] = np.full((198, 2), np.nan, dtype=np.float32)
        if not 0 <= int(frame) < 198:
            raise ValueError("Invalid frame index")
        result[key][int(frame)] = [float(first), float(second)]
    if not result or any(not np.isfinite(matrix).all() for matrix in result.values()):
        raise ValueError("Missing per-frame FVP outputs")
    return result


def reference_windows(program: Path, samples: Path) -> dict:
    """Execute the reference graph with the firmware's PCM and window policy.

    :param program: PT2E int8.pt2 exported with the PTE.
    :param samples: Repository WAV directory in firmware filename order.
    :returns:       Reference matrices keyed by clip and window start.
    """
    importlib.import_module("torchao.quantization.pt2e.quantize_pt2e")
    model = torch.export.load(program).module()
    result = {}
    for clip, path in enumerate(sorted(samples.glob("*.wav"))):
        with wave.open(str(path)) as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
                raise ValueError(f"Expected 16 kHz mono PCM16: {path}")
            pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").copy()
        pcm = np.pad(pcm, (0, max(0, 31920 - len(pcm))))
        for start in range(0, len(pcm), 16000):
            signal = torch.from_numpy(pcm[start:start + 31920]).float()
            signal = torch.nn.functional.pad(signal, (0, 31920 - signal.numel()))
            features = torchaudio.compliance.kaldi.fbank(
                signal[None], num_mel_bins=40, frame_length=25, frame_shift=10,
                dither=0.0, energy_floor=0.0, sample_frequency=16000)
            with torch.no_grad():
                result[(clip, start)] = model(features[None]).numpy()[0]
            if start + 31920 >= len(pcm):
                break
    return result


def main():
    """Check complete windows, peak decisions and mean numerical error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-mae", type=float, default=0.005)
    parser.add_argument("--max-peak-error", type=float, default=0.05)
    args = parser.parse_args()
    torch.set_num_threads(4)
    actual = read_scores(args.log)
    expected = reference_windows(args.program, args.samples)
    if set(actual) != set(expected):
        raise ValueError("FVP and reference inference windows differ")
    rows = []
    for key, reference in expected.items():
        error = np.abs(actual[key] - reference)
        peaks = actual[key].max(axis=0)
        reference_peaks = reference.max(axis=0)
        rows.append({"clip": key[0], "start_sample": key[1], "mae": float(error.mean()),
                     "max_abs": float(error.max()), "fvp_peaks": peaks.tolist(),
                     "reference_peaks": reference_peaks.tolist(),
                     "max_peak_error": float(np.abs(peaks - reference_peaks).max()),
                     "threshold_disagreements": int(np.count_nonzero(
                         (peaks >= .5) != (reference_peaks >= .5)))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    if any(row["mae"] > args.max_mae or row["max_peak_error"] > args.max_peak_error
           or row["threshold_disagreements"] for row in rows):
        raise ValueError("FVP outputs exceed the configured reference tolerance")


if __name__ == "__main__":
    main()
