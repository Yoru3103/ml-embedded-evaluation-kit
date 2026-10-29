# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Conservative offline denoising using energy-gated head/tail noise candidates."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
FRAME_SIZE = 512
HOP_SIZE = 160


@dataclass
class NoiseOptions:
    """Host-side profile selection and attenuation limits."""
    threshold_dbfs: float = -40.0
    edge_seconds: float = 1.0
    minimum_seconds: float = 0.2
    source: str = "auto"
    reduction_db: float = 6.0


def consecutive_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs of true frame indices."""
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def select_noise(frames: np.ndarray, starts: np.ndarray, samples: int,
                 options: NoiseOptions) -> tuple[np.ndarray, dict]:
    """Select sustained quiet runs fully contained within the requested edges.

    :param frames:  Unwindowed analysis frames in full-scale floating point.
    :param starts:  Frame start offsets, including analysis padding.
    :param samples: Original recording length in samples.
    :param options: Selection thresholds and edge policy.
    :returns:       Selected frame mask and measured selection details.
    """
    levels = 20 * np.log10(np.maximum(np.sqrt(np.mean(frames ** 2, axis=1)), 1e-12))
    peaks = 20 * np.log10(np.maximum(np.max(np.abs(frames), axis=1), 1e-12))
    edge_samples = round(options.edge_seconds * SAMPLE_RATE)
    ends = starts + FRAME_SIZE
    edges = {"head": ends <= edge_samples,
             "tail": starts >= samples - edge_samples,
             "auto": (ends <= edge_samples) | (starts >= samples - edge_samples)}[options.source]
    eligible = ((starts >= 0) & (ends <= samples) & edges &
                (levels < options.threshold_dbfs) & (peaks < options.threshold_dbfs + 12))
    selected = np.zeros(len(frames), dtype=bool)
    regions = []
    for first, end in consecutive_ranges(eligible):
        length = (end - first - 1) * HOP_SIZE + FRAME_SIZE
        if length >= options.minimum_seconds * SAMPLE_RATE:
            selected[first:end] = True
            regions.append({"start_seconds": float(starts[first] / SAMPLE_RATE),
                            "end_seconds": float(ends[end - 1] / SAMPLE_RATE),
                            "median_rms_dbfs": float(np.median(levels[first:end]))})
    return selected, {"selected_frames": int(selected.sum()), "regions": regions,
                      "threshold_dbfs": options.threshold_dbfs,
                      "peak_guard_dbfs": options.threshold_dbfs + 12,
                      "source": options.source, "edge_seconds": options.edge_seconds,
                      "minimum_seconds": options.minimum_seconds}


def smooth_gains(gains: np.ndarray) -> np.ndarray:
    """Smooth adjacent frequency bins and time frames without shifting the audio."""
    frequency = np.pad(gains, ((0, 0), (1, 1)), mode="edge")
    result = (frequency[:, :-2] + 2 * frequency[:, 1:-1] + frequency[:, 2:]) / 4
    temporal = np.pad(result, ((1, 1), (0, 0)), mode="edge")
    return (temporal[:-2] + 2 * temporal[1:-1] + temporal[2:]) / 4


def overlap_add(frames: np.ndarray, window: np.ndarray, samples: int) -> np.ndarray:
    """Reconstruct the waveform and remove only artificial analysis padding.

    :param frames:  Inverse-transformed, overlapping audio frames.
    :param window:  Synthesis window, matching the analysis window.
    :param samples: Original sample count to preserve.
    :returns:       Aligned waveform with its original length.
    """
    output = np.zeros((len(frames) - 1) * HOP_SIZE + FRAME_SIZE)
    weights = np.zeros(len(output))
    for index, frame in enumerate(frames):
        begin = index * HOP_SIZE
        output[begin:begin + FRAME_SIZE] += frame * window
        weights[begin:begin + FRAME_SIZE] += window ** 2
    output /= np.maximum(weights, 1e-12)
    return output[FRAME_SIZE // 2:FRAME_SIZE // 2 + samples]


def denoise(audio: np.ndarray, options: NoiseOptions) -> tuple[np.ndarray, dict]:
    """Attenuate a stationary noise estimate while preserving length and alignment.

    :param audio:   Mono audio scaled to the interval [-1, 1).
    :param options: Energy gate and maximum attenuation settings.
    :returns:       Processed audio and a report; unsuitable profiles are skipped.
    """
    padded = np.pad(audio, (FRAME_SIZE // 2, FRAME_SIZE // 2 + (-len(audio)) % HOP_SIZE),
                    mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, FRAME_SIZE)[::HOP_SIZE]
    starts = np.arange(len(frames)) * HOP_SIZE - FRAME_SIZE // 2
    selected, report = select_noise(frames, starts, len(audio), options)
    report["max_reduction_db"] = options.reduction_db
    if not np.any(selected):
        return audio.copy(), {**report, "applied": False,
                              "reason": "No sustained quiet edge region"}
    window = np.sqrt(np.hanning(FRAME_SIZE + 1)[:-1])
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = np.abs(spectrum) ** 2
    noise = np.median(power[selected], axis=0)
    floor = 10 ** (-options.reduction_db / 20)
    gains = np.sqrt(np.clip(1 - noise / np.maximum(power, 1e-20), floor ** 2, 1))
    reconstructed = np.fft.irfft(spectrum * smooth_gains(gains), n=FRAME_SIZE, axis=1)
    output = overlap_add(reconstructed, window, len(audio))
    report["clipped_samples"] = int(np.count_nonzero((output < -1) | (output > 32767 / 32768)))
    return np.clip(output, -1, 32767 / 32768), {**report, "applied": True}


def parse_args() -> argparse.Namespace:
    """Parse and validate the independent WAV experiment options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path("build/asr-denoise"))
    parser.add_argument("--noise-threshold-dbfs", type=float, default=-40)
    parser.add_argument("--noise-source", choices=("auto", "head", "tail"), default="auto")
    parser.add_argument("--edge-seconds", type=float, default=1)
    parser.add_argument("--minimum-noise-seconds", type=float, default=0.2)
    parser.add_argument("--max-reduction-db", type=float, default=6)
    args = parser.parse_args()
    if not (-100 <= args.noise_threshold_dbfs <= -20 and
            0.2 <= args.edge_seconds <= 5 and
            0.05 <= args.minimum_noise_seconds <= args.edge_seconds and
            0 <= args.max_reduction_db <= 12):
        parser.error("Invalid gate, edge duration, minimum duration or attenuation limit")
    return args


def main():
    """Write a processed copy and reproducible profile-selection report."""
    args = parse_args()
    info = sf.info(args.input)
    if (info.samplerate != SAMPLE_RATE or info.channels != 1 or info.subtype != "PCM_16" or
            info.format not in ("WAV", "WAVEX") or not 512 <= info.frames <= 480000):
        raise ValueError("Requires a 16 kHz mono PCM16 WAV, 512..480000 samples")
    directory = args.output_directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    original = directory / "original.wav"
    destination = directory / "denoised.wav"
    if args.input.resolve() in (original, destination):
        raise ValueError("Choose an output directory that does not contain the input as an output")
    audio, _ = sf.read(args.input, dtype="float64")
    options = NoiseOptions(args.noise_threshold_dbfs, args.edge_seconds,
                           args.minimum_noise_seconds, args.noise_source, args.max_reduction_db)
    started = time.perf_counter()
    processed, report = denoise(audio, options)
    report["denoise_host_ms"] = (time.perf_counter() - started) * 1000
    shutil.copyfile(args.input, original)
    if report["applied"]:
        sf.write(destination, processed, SAMPLE_RATE, subtype="PCM_16")
    else:
        shutil.copyfile(args.input, destination)
    report.update({"input": str(args.input.resolve()), "samples": len(audio),
                   "seconds": len(audio) / SAMPLE_RATE, "output": str(destination),
                   "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                   "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                   "caution": "Energy gating is a noise-profile heuristic, not speech detection"})
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
