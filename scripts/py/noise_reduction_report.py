# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Generate an offline A/B listening and measurement report for RNNoise WAVs."""

import argparse
from array import array
import base64
import html
import io
import json
import math
from pathlib import Path
import sys
import wave


SAMPLE_RATE = 48000
FULL_SCALE = 32768


def read_pcm(path: Path) -> array:
    """
    Read a mono PCM16 WAV at the firmware sample rate.

    :param path:    Input WAV path.
    :returns:       Signed PCM samples.
    """
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (
                1, 2, SAMPLE_RATE):
            raise ValueError(f"{path}: expected 48 kHz mono PCM16 WAV")
        frames = source.getnframes()
        samples = array("h")
        samples.frombytes(source.readframes(frames))
    if len(samples) != frames or not samples:
        raise ValueError(f"{path}: empty or truncated WAV")
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def db_ratio(numerator: float, denominator: float) -> float | None:
    """
    Convert a power ratio to decibels.

    :param numerator:   Signal power.
    :param denominator: Reference power.
    :returns:           Decibels, or None for a zero power term.
    """
    if numerator <= 0 or denominator <= 0:
        return None
    return 10 * math.log10(numerator / denominator)


def rms_dbfs(samples: array) -> float | None:
    """
    Measure RMS relative to full scale.

    :param samples: PCM samples.
    :returns:       Level in dBFS, or None for digital silence.
    """
    return db_ratio(sum(value * value for value in samples), len(samples) * FULL_SCALE**2)


def si_sdr(samples: array, clean: array) -> float | None:
    """
    Measure SI-SDR with zero-mean signals on the same sample interval.

    :param samples:     Estimated speech samples.
    :param clean:       Time-aligned clean speech.
    :returns:           SI-SDR in dB, or None for a degenerate comparison.
    """
    sample_mean = sum(samples) / len(samples)
    clean_mean = sum(clean) / len(clean)
    clean_energy = math.fsum((value - clean_mean)**2 for value in clean)
    if not clean_energy:
        return None
    scale = math.fsum((value - sample_mean) * (ref - clean_mean)
                      for value, ref in zip(samples, clean)) / clean_energy
    distortion = math.fsum((value - sample_mean - scale * (ref - clean_mean))**2
                           for value, ref in zip(samples, clean))
    return db_ratio(scale**2 * clean_energy, distortion)


def summarize(samples: array) -> dict:
    """
    Compute the level, digital rail hits and plot envelopes.

    :param samples: PCM samples in the common interval.
    :returns:       Levels and downsampled plot data.
    """
    width = min(1000, len(samples))
    bins = []
    for index in range(width):
        chunk = samples[index * len(samples) // width:(index + 1) * len(samples) // width]
        bins.append((min(chunk), max(chunk), rms_dbfs(chunk)))
    return {
        "rms_dbfs": rms_dbfs(samples),
        "peak_dbfs": db_ratio(max(abs(min(samples)), abs(max(samples)))**2, FULL_SCALE**2),
        "rail_hit_samples": samples.count(-FULL_SCALE) + samples.count(FULL_SCALE - 1),
        "bins": bins,
    }


def audio_source(samples: array) -> str:
    """
    Encode the aligned samples as an embedded WAV without changing their level.

    :param samples: PCM samples in host byte order.
    :returns:       Base64 data for the browser audio control.
    """
    if sys.byteorder == "big":
        samples = samples[:]
        samples.byteswap()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        output.writeframes(samples.tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def render_report(tracks: dict[str, array], summaries: dict, metrics: dict) -> str:
    """
    Render a self-contained browser report.

    :param tracks:      Aligned PCM tracks by role.
    :param summaries:   Level and plot data by role.
    :param metrics:     Measurement metadata and results.
    :returns:           Report HTML.
    """
    players = "".join(
        f'<div>{role}<audio aria-label="{role}" controls preload="metadata" '
        f'src="data:audio/wav;base64,{audio_source(samples)}"></audio></div>'
        for role, samples in tracks.items()
    )
    rows = ""
    for role, values in metrics["tracks"].items():
        cells = "".join(
            f"<td>{value:.2f}</td>" if isinstance(value, float)
            else f"<td>{value if value is not None else '—'}</td>"
            for value in (values["rms_dbfs"], values["peak_dbfs"], values["rail_hit_samples"],
                          values.get("si_sdr_db"))
        )
        rows += f"<tr><th scope='row'>{role}</th>{cells}</tr>"
    delta = metrics.get("si_sdri_db")
    improvement = f"<p>SI-SDR improvement: {delta:+.2f} dB</p>" if delta is not None else ""
    sources = " · ".join(f"{role}: {html.escape(values['source'])}"
                         for role, values in metrics["tracks"].items())
    plot_data = json.dumps({role: summary["bins"] for role, summary in summaries.items()})
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Noise reduction evaluation</title>
<style>
:root {{ color-scheme: light; font: 16px/1.5 'Instrument Sans', system-ui, sans-serif; }}
body {{ max-width: 1080px; padding: 1.5rem; margin: auto; color: #142c34; background: #f7faf9; }}
h1 {{ margin-bottom: .25rem; }} p {{ overflow-wrap: anywhere; }}
.players {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
audio {{ display: block; width: min(300px, 78vw); }}
table {{ border-collapse: collapse; width: 100%; margin-block: 1.25rem; text-align: right; }}
th, td {{ padding: .45rem; border-bottom: 1px solid #c6d6d5; }}
th:first-child {{ text-align: left; }}
canvas {{ width: 100%; height: auto; display: block; background: white; }}
small {{ color: #3c5159; }}
</style>
<main>
<h1>Noise reduction evaluation</h1>
<p>{metrics['duration_seconds']:.3f} s common interval · 48 kHz mono ·
   Denoised delay {metrics['denoised_delay_samples']:+d} samples</p>
<p><small>{sources}</small></p>
<div class="players">{players}</div>
<p><button id="compare" type="button">Switch between noisy and denoised</button></p>
<table><thead><tr><th>Track</th><th>RMS (dBFS)</th><th>Peak (dBFS)</th>
<th>Rail hits</th><th>SI-SDR (dB)</th></tr></thead><tbody>{rows}</tbody></table>
{improvement}
<small>RMS reflects overall signal level. SI-SDR requires aligned clean speech.
A dash denotes an unavailable value.</small>
<h2>Waveform</h2>
<canvas id="waveform" width="1080" height="265" aria-label="Aligned waveform envelopes"></canvas>
<h2>Short-time RMS</h2>
<canvas id="rms" width="1080" height="265" aria-label="Short-time RMS in dBFS"></canvas>
</main>
<script>
const tracks = {plot_data};
const seconds = {metrics['duration_seconds']};
const colors = ["#3166a7", "#007764", "#a45724"];
const players = [...document.querySelectorAll("audio")];
let active = 0;
players.forEach((player, index) => player.addEventListener("play", () => {{
  active = index;
  players.forEach(other => {{ if (other !== player) other.pause(); }});
}}));
document.querySelector("#compare").addEventListener("click", async () => {{
  const time = players[active].currentTime;
  active = active === 0 ? 1 : 0;
  const player = players[active];
  player.currentTime = time;
  await player.play();
}});

function draw(id, ticks, range, waveform) {{
  const canvas = document.getElementById(id);
  const ctx = canvas.getContext("2d");
  const left = 56, right = canvas.width - 22, top = 32, bottom = canvas.height - 34;
  const y = value => bottom - (value - range[0]) / (range[1] - range[0]) * (bottom - top);
  ctx.font = "13px system-ui";
  ctx.strokeStyle = "#dbe6e5";
  ctx.fillStyle = "#34525a";
  for (const value of ticks) {{
    ctx.beginPath(); ctx.moveTo(left, y(value)); ctx.lineTo(right, y(value)); ctx.stroke();
    ctx.fillText(value, 7, y(value) + 4);
  }}
  for (let tick = 0; tick <= 5; ++tick) {{
    const x = left + (right - left) * tick / 5;
    ctx.fillText((seconds * tick / 5).toFixed(2) + " s", x - 20, bottom + 22);
  }}
  Object.entries(tracks).forEach(([name, bins], index) => {{
    ctx.fillStyle = ctx.strokeStyle = colors[index];
    ctx.fillText(name, left + index * 110, 19);
    ctx.globalAlpha = waveform ? .55 : 1;
    ctx.beginPath();
    bins.forEach(([low, high, rms], i) => {{
      const x = left + (right - left) * (i + .5) / bins.length;
      if (waveform) {{
        ctx.moveTo(x, y(low / 32768)); ctx.lineTo(x, y(high / 32768));
      }} else {{
        const level = y(Math.max(-100, rms ?? -100));
        if (i === 0) ctx.moveTo(x, level); else ctx.lineTo(x, level);
      }}
    }});
    ctx.stroke();
    ctx.globalAlpha = 1;
  }});
}}
draw("waveform", [-1, -.5, 0, .5, 1], [-1, 1], true);
draw("rms", [-100, -75, -50, -25, 0], [-100, 0], false);
</script></html>
"""


def main():
    """Read the input tracks and write a report and its JSON metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noisy", type=Path, required=True)
    parser.add_argument("--denoised", type=Path, required=True)
    parser.add_argument("--clean", type=Path,
                        help="Optional clean reference aligned to noisy audio.")
    parser.add_argument("--denoised-delay-samples", type=int, default=512,
                        help="Positive when denoised audio lags noisy audio (RNNoise: 512).")
    parser.add_argument("--output-dir", type=Path, default=Path("noise-reduction-report"))
    args = parser.parse_args()

    paths = {"Noisy": args.noisy, "Denoised": args.denoised}
    if args.clean:
        paths["Clean"] = args.clean
    try:
        tracks = {role: read_pcm(path) for role, path in paths.items()}
        delay = args.denoised_delay_samples
        offsets = {role: max(0, delay if role == "Denoised" else -delay) for role in tracks}
        length = min(len(samples) - offsets[role] for role, samples in tracks.items())
        if length <= 0:
            raise ValueError("Audio tracks have no common interval at the specified delay")
        tracks = {role: samples[offsets[role]:offsets[role] + length]
                  for role, samples in tracks.items()}
        summaries = {role: summarize(samples) for role, samples in tracks.items()}
        metrics = {
            "sample_rate_hz": SAMPLE_RATE,
            "compared_samples": length,
            "duration_seconds": length / SAMPLE_RATE,
            "denoised_delay_samples": delay,
            "tracks": {
                role: {"source": str(paths[role]), "start_sample": offsets[role],
                       **{key: value for key, value in summary.items() if key != "bins"}}
                for role, summary in summaries.items()
            },
        }
        if args.clean:
            scores = [si_sdr(tracks[role], tracks["Clean"]) for role in ("Noisy", "Denoised")]
            for role, score in zip(("Noisy", "Denoised"), scores):
                metrics["tracks"][role]["si_sdr_db"] = score
            before, after = scores
            metrics["si_sdri_db"] = (
                after - before if before is not None and after is not None else None
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = args.output_dir / "report.html"
        report.write_text(render_report(tracks, summaries, metrics), encoding="utf-8")
        output = args.output_dir / "metrics.json"
        output.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (EOFError, OSError, ValueError, wave.Error) as error:
        parser.exit(1, f"{error}\n")
    print(report.resolve())
    print(output.resolve())


if __name__ == "__main__":
    main()
