# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Compare the production C++ Fbank frontend with torchaudio's Kaldi reference."""
import argparse
import json
from pathlib import Path
import subprocess
import wave

import numpy as np
import torch
import torchaudio


def compare(executable: Path, sample: Path, output: Path) -> dict:
    """Compare one repository WAV and save both feature matrices.

    :param executable:  Native processing-test executable.
    :param sample:      PCM16, 16 kHz, mono WAV.
    :param output:      Directory for raw data and numerical results.
    :returns:           Error statistics and generated file paths.
    """
    with wave.open(str(sample)) as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"Expected 16 kHz mono PCM16: {sample}")
        pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").copy()
    pcm_path = output / (sample.stem + ".pcm")
    cpp_path = output / (sample.stem + "_cpp.f32")
    pcm.tofile(pcm_path)
    subprocess.run([str(executable), "--features", str(pcm_path), str(cpp_path)], check=True)
    signal = torch.from_numpy(pcm[:31920]).float()
    signal = torch.nn.functional.pad(signal, (0, max(0, 31920 - signal.numel())))
    reference = torchaudio.compliance.kaldi.fbank(
        signal[None], num_mel_bins=40, frame_length=25, frame_shift=10,
        dither=0.0, energy_floor=0.0, sample_frequency=16000).numpy()
    actual = np.fromfile(cpp_path, dtype="<f4").reshape(198, 40)
    error = np.abs(actual - reference)
    reference_path = output / (sample.stem + "_reference.npy")
    np.save(reference_path, reference)
    np.save(output / (sample.stem + "_cpp.npy"), actual)
    return {"sample": str(sample), "mae": float(error.mean()), "max_abs": float(error.max()),
            "p99_abs": float(np.quantile(error, .99)), "reference": str(reference_path),
            "cpp": str(cpp_path)}


def main():
    """Run the repository sample comparisons and enforce the frontend tolerance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-error", type=float, default=0.02)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    samples = sorted(args.samples.glob("*.wav"))
    if not samples:
        raise ValueError("No repository WAV samples selected")
    results = [compare(args.executable, sample, args.output) for sample in samples]
    (args.output / "fbank_comparison.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    if any(row["max_abs"] > args.max_error for row in results):
        raise ValueError(f"Frontend error exceeds {args.max_error}")


if __name__ == "__main__":
    main()
