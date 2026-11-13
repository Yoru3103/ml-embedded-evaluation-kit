# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Compare RNNoise FVP outputs against a paired VoiceBank-DEMAND test set."""

import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics
import sys
import wave

from noise_reduction_report import SAMPLE_RATE, read_pcm, si_sdr


def dbfs(samples) -> float | None:
    """
    Measure an audio segment's RMS relative to PCM16 full scale.

    :param samples: PCM16 samples.
    :returns:       RMS dBFS, or None for digital silence.
    """
    power = math.fsum(value * value for value in samples) / len(samples)
    if power == 0:
        return None
    return 10 * math.log10(power / (32768**2))


def load_frames(path: Path) -> int:
    """
    Validate a WAV's format and return its frame count.

    :param path:     WAV file path.
    :returns:        Number of frames.
    :raises ValueError: If the file format differs from the firmware's PCM format.
    """
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError(f"{path}: expected 48 kHz mono PCM16 WAV")
        return audio.getnframes()


def make_row(noisy_path: Path, clean_path: Path, output_path: Path,
             delay_samples: int) -> dict:
    """
    Measure SI-SDR before and after denoising on the shared aligned samples.

    :param noisy_path:      Noisy test utterance.
    :param clean_path:      Paired clean reference utterance.
    :param output_path:     Corresponding denoised FVP output.
    :param delay_samples:   Number of leading samples to skip in the output.
    :returns:               Per-utterance measurements.
    """
    noisy = read_pcm(noisy_path)
    clean = read_pcm(clean_path)
    denoised = read_pcm(output_path)
    length = min(len(noisy), len(clean), len(denoised) - delay_samples)
    if length <= 0:
        raise ValueError(f"{output_path.name}: no common samples after alignment")
    noisy = noisy[:length]
    clean = clean[:length]
    denoised = denoised[delay_samples:delay_samples + length]
    before = si_sdr(noisy, clean)
    after = si_sdr(denoised, clean)
    if before is None or after is None:
        raise ValueError(f"{clean_path.name}: SI-SDR is undefined for the reference")
    return {
        "filename": noisy_path.name,
        "samples": length,
        "duration_seconds": length / SAMPLE_RATE,
        "si_sdr_noisy_db": before,
        "si_sdr_denoised_db": after,
        "si_sdri_db": after - before,
        "rms_noisy_dbfs": dbfs(noisy),
        "rms_denoised_dbfs": dbfs(denoised),
    }


def render_html(rows: list[dict], summary: dict) -> str:
    """
    Render a standalone per-utterance results page.

    :param rows:     Results in the original dataset's filename order.
    :param summary:  Aggregated scores.
    :returns:        UTF-8 HTML.
    """
    values = json.dumps([row["si_sdri_db"] for row in rows])
    table = "".join(
        "<tr><td>{filename}</td><td>{before:.2f}</td><td>{after:.2f}</td>"
        "<td>{improvement:+.2f}</td><td>{duration:.2f}</td></tr>".format(
            filename=html.escape(row["filename"]),
            before=row["si_sdr_noisy_db"],
            after=row["si_sdr_denoised_db"],
            improvement=row["si_sdri_db"],
            duration=row["duration_seconds"],
        )
        for row in rows
    )
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VoiceBank-DEMAND noise reduction results</title>
<style>
body {{ max-width: 1080px; margin: auto; padding: 24px; background: #f7faf9;
       color: #142c34; font: 16px/1.5 'Instrument Sans', system-ui, sans-serif; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 14px; }}
.card {{ background: white; border: 1px solid #c6d6d5; border-radius: 12px;
         padding: 14px 18px; min-width: 175px; }}
.value {{ display: block; font-size: 1.55rem; font-weight: 650; }}
canvas {{ width: 100%; background: white; display: block; }}
.table-wrap {{ overflow: auto; max-height: 70vh; background: white; }}
table {{ border-collapse: collapse; width: 100%; text-align: right; }}
th, td {{ padding: 7px 10px; border-bottom: 1px solid #dbe6e5; }}
th {{ position: sticky; top: 0; background: #edf4f2; }}
th:first-child, td:first-child {{ text-align: left; }}
</style>
<main>
<h1>VoiceBank-DEMAND test results</h1>
<p>48 kHz mono PCM16 · {summary['utterance_count']} paired utterances ·
{summary['total_audio_seconds'] / 60:.1f} minutes scored ·
{summary['denoised_delay_samples']} sample output delay removed</p>
<div class="cards">
<div class="card">Mean SI-SDR<input class="value" readonly value="{summary['mean_si_sdr_noisy_db']:.2f} → {summary['mean_si_sdr_denoised_db']:.2f} dB"></div>
<div class="card">Mean SI-SDR improvement<span class="value">{summary['mean_si_sdri_db']:+.2f} dB</span></div>
<div class="card">Median SI-SDR improvement<span class="value">{summary['median_si_sdri_db']:+.2f} dB</span></div>
<div class="card">Utterances improved<span class="value">{summary['improved_count']} / {summary['utterance_count']}</span></div>
</div>
<h2>SI-SDR improvement distribution</h2>
<canvas id="histogram" width="1080" height="300"></canvas>
<h2>Per-utterance scores</h2>
<div class="table-wrap"><table><thead><tr><th>Utterance</th>
<th>Noisy SI-SDR (dB)</th><th>Denoised SI-SDR (dB)</th>
<th>Improvement (dB)</th><th>Seconds</th></tr></thead><tbody>{table}</tbody></table></div>
</main>
<script>
const scores = {values};
const canvas = document.querySelector("#histogram"), ctx = canvas.getContext("2d");
const low = Math.floor(Math.min(...scores)), high = Math.ceil(Math.max(...scores));
const count = Math.max(1, high - low + 1), bins = Array(count).fill(0);
scores.forEach(value => bins[Math.min(count - 1, Math.floor(value) - low)]++);
const left = 55, right = canvas.width - 20, top = 28, bottom = canvas.height - 42;
const peak = Math.max(...bins, 1);
ctx.font = "13px system-ui"; ctx.fillStyle = "#3166a7"; ctx.strokeStyle = "#dbe6e5";
for (let tick = 0; tick <= 4; ++tick) {{
  const y = bottom - (bottom - top) * tick / 4;
  ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
  ctx.fillStyle = "#34525a"; ctx.fillText(Math.round(peak * tick / 4), 10, y + 4);
}}
bins.forEach((value, index) => {{
  const width = (right-left)/count;
  const height = (bottom-top)*value/peak;
  ctx.fillStyle = "#3166a7";
  ctx.fillRect(left + index*width + 1, bottom-height, Math.max(1,width-2), height);
  if (index % Math.max(1, Math.floor(count/12)) === 0) {{
    ctx.fillStyle = "#34525a"; ctx.fillText((low+index) + " dB", left+index*width, bottom+21);
  }}
}});
</script></html>
"""


def main():
    """Validate the three input directories and write CSV, JSON and HTML results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--denoised-dir", type=Path, required=True)
    parser.add_argument("--denoised-delay-samples", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        noisy_files = sorted(args.noisy_dir.glob("*.wav"))
        clean_files = sorted(args.clean_dir.glob("*.wav"))
        if not noisy_files or len(noisy_files) != len(clean_files):
            raise ValueError(
                f"WAV count mismatch: noisy={len(noisy_files)}, clean={len(clean_files)}"
            )
        if [path.name for path in noisy_files] != [path.name for path in clean_files]:
            raise ValueError("Noisy and clean filenames do not match in sorted order")
        named_outputs = [args.denoised_dir / f"denoised_{path.name}" for path in noisy_files]
        loop_outputs = sorted(
            args.denoised_dir.glob("denoised_loop*.wav"),
            key=lambda path: int(path.stem.removeprefix("denoised_loop")),
        )
        if all(path.is_file() for path in named_outputs) and not loop_outputs:
            output_files = named_outputs
        elif not any(path.is_file() for path in named_outputs) and len(loop_outputs) == len(noisy_files):
            output_files = loop_outputs
        else:
            raise ValueError(
                "Denoised outputs must be either one named WAV per utterance or a complete "
                "denoised_loop0.wav through denoised_loopN.wav set"
            )
        missing_outputs = [path.name for path in output_files if not path.is_file()]
        if missing_outputs:
            raise ValueError(
                f"Missing {len(missing_outputs)} denoised outputs, first is {missing_outputs[0]}"
            )
        for path in noisy_files + clean_files:
            load_frames(path)

        rows = [
            make_row(noisy, clean, denoised, args.denoised_delay_samples)
            for noisy, clean, denoised in zip(noisy_files, clean_files, output_files)
        ]
        improvements = [row["si_sdri_db"] for row in rows]
        summary = {
            "dataset": "VoiceBank-DEMAND test set",
            "sample_rate_hz": SAMPLE_RATE,
            "denoised_delay_samples": args.denoised_delay_samples,
            "utterance_count": len(rows),
            "total_audio_seconds": sum(row["duration_seconds"] for row in rows),
            "mean_si_sdr_noisy_db": statistics.fmean(row["si_sdr_noisy_db"] for row in rows),
            "mean_si_sdr_denoised_db": statistics.fmean(row["si_sdr_denoised_db"] for row in rows),
            "mean_si_sdri_db": statistics.fmean(improvements),
            "median_si_sdri_db": statistics.median(improvements),
            "p10_si_sdri_db": sorted(improvements)[math.floor(0.1 * (len(rows) - 1))],
            "p90_si_sdri_db": sorted(improvements)[math.floor(0.9 * (len(rows) - 1))],
            "improved_count": sum(value > 0 for value in improvements),
            "degraded_count": sum(value < 0 for value in improvements),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "dataset_metrics.json"
        json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        csv_path = args.output_dir / "per_utterance_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        html_path = args.output_dir / "dataset_report.html"
        html_path.write_text(render_html(rows, summary), encoding="utf-8")
    except (OSError, ValueError, wave.Error, EOFError) as error:
        parser.exit(1, f"{error}\n")

    print(json.dumps(summary, indent=2))
    print(f"HTML report: {html_path.resolve()}")
    print(f"Per-utterance CSV: {csv_path.resolve()}")
    print(f"Summary JSON: {json_path.resolve()}")


if __name__ == "__main__":
    main()
