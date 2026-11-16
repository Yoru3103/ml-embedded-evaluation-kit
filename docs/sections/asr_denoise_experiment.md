# Host-side ASR denoising experiment

The normal sender now supports this processing directly with `-Denoise`:

```powershell
./scripts/windows/send-asr-audio.ps1 -Port COM5 -Denoise `
    -WavPath "C:\audio\recording.wav"
```

It also works with `-TcpHost`/`-TcpPort`. NoiseSource, NoiseThresholdDbfs,
EdgeSeconds and MaxReductionDb accept the same options described below.
`-Denoise -ValidateOnly` processes and validates without connecting to a target.
The input remains unchanged, temporary outputs are removed, and the transmitted
PCM CRC is computed after processing. A missing noise profile bypasses processing.
The sender prints Denoise applied/skipped, Denoise.Host (including Python startup,
file I/O and processed-WAV loading), and Total.Wall (including preprocessing,
connection, handshake and the request; excluding result console printing).
Request.Wall and target End-to-end retain their previous definitions.

Only `-Denoise` requires Python with numpy and soundfile. The repository environment
is selected automatically; `-DenoisePython` can specify another interpreter.
The FPGA package includes the helper and Python script, but not the Python runtime.

This independent experiment creates original/processed WAV copies and a JSON
report. It does not upload audio, invoke a board, or modify the source recording.
It uses the existing Windows ExecuTorch Python environment's numpy and soundfile.

```powershell
./scripts/windows/test-asr-denoise.ps1 `
    -WavPath "C:\audio\recording.wav" `
    -OutputDirectory build/asr-denoise/another-door-self
```

The output directory contains `original.wav`, `denoised.wav` and `report.json`.
The report records selected noise intervals, energy levels, applied/skipped state,
host processing time, sample count and input/output SHA256. Files retain their
sample count and alignment. Input must be 16 kHz mono PCM16 WAV, up to 30 seconds.

## Profile selection

- `-NoiseSource auto` examines both ends; `head` or `tail` restricts the search.
- `-EdgeSeconds 1` searches the first/last second by default.
- `-NoiseThresholdDbfs -40` accepts only frames with RMS below -40 dBFS. This is
  level relative to digital full scale, not acoustic sound pressure level.
- A second peak gate rejects frames with peaks at or above threshold + 12 dB.
- Eligible regions must remain quiet for at least 0.2 seconds. Frames are 512
  samples long with a 160-sample hop; padded analysis frames cannot form a profile.
- No eligible region means an exact file-copy bypass, not forced denoising.

For example, to estimate noise only from quiet portions of the tail:

```powershell
./scripts/windows/test-asr-denoise.ps1 -WavPath "./recording.wav" `
    -NoiseSource tail -NoiseThresholdDbfs -45 -EdgeSeconds 1.5 `
    -MaxReductionDb 6 -OutputDirectory build/asr-denoise/tail
```

Energy gating is a heuristic, not VAD: quiet speech can pass and loud stationary
noise can be rejected. Do not raise the gate merely to force a profile. The
method uses a median noise-power spectrum, a frequency/time-smoothed spectral
gain and overlap-add reconstruction. Maximum attenuation defaults to 6 dB.
It performs no normalization, resampling, leading/trailing trimming or accent
correction. The original recording is retained for comparison.

## Tests

```powershell
./resources_downloaded/windows-executorch/env/Scripts/python.exe -B `
    tests/processing/asr/test_denoise_asr_wav.py
```

Synthetic checks cover a loud start with a quiet tail, explicit head-only
rejection, exact bypass when no profile qualifies, rejection of brief gaps,
zero-attenuation reconstruction, sample alignment and all-zero input.
