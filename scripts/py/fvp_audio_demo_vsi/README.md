# FVP audio pull bridge

Start the host from the repository root:

```powershell
resources_downloaded/windows-executorch/env/Scripts/python.exe scripts/py/fvp_audio_demo.py --wav speech.wav --queue-ms 1000
```

WAV input must be uncompressed mono 16 kHz signed PCM16. `--realtime` optionally
paces WAV acquisition. `--duration SECONDS` limits either source. WAV reads use
backpressure and never discard samples. Both sources read at most 320 samples
at a time. The queue holds at most `floor(queue_ms / 20)` blocks; default 1000 ms.

Microphone support is optional:

```powershell
resources_downloaded/windows-executorch/env/Scripts/python.exe -m pip install sounddevice
resources_downloaded/windows-executorch/env/Scripts/python.exe scripts/py/fvp_audio_demo.py --list-devices
resources_downloaded/windows-executorch/env/Scripts/python.exe scripts/py/fvp_audio_demo.py --microphone --device 1 --queue-ms 1000
```

Enumeration does not open a capture stream. The server opens no source until the
first `NEXT` request. Microphone input requires native 16 kHz mono PCM16 support;
unsupported devices fail explicitly. Capture runs in a separate producer thread.
When full, its queue drops oldest blocks and marks the next surviving interval
`DISCONTINUITY`. Backend overflow also marks discontinuity; sample offsets count
acquired samples, since the backend does not report an exact lost-sample count.
Sentence VAD and recognition run on the target.

Set the FVP `mps4_board.v_path` to this directory's absolute path. No dependency
or HAL modifications are needed. This adapter replaces the VSI0 Python module;
it does not implement the CMSIS-vStream driver protocol. Its standard-library
Python code uses Python 3.9-compatible syntax for embedded FVP interpreters.

## Connection protocol

The single-client host listens on `127.0.0.1:6010`, using
`multiprocessing.connection.Listener` with auth key `b"fvp_audio_demo"`.
`--port` overrides the host port; set `FVP_AUDIO_DEMO_PORT` in the FVP environment
to the same value. Each `send_bytes(b"NEXT")` receives a fixed 664-byte response
through `recv_bytes(664)`. No pickle object transport is used.

The header is `struct.Struct("<IQIII")`, followed by 640 bytes of little-endian
PCM16, zero-padded after the valid sample count:

| Field | Type | Meaning |
| --- | --- | --- |
| magic | uint32 | `0x41554431` |
| firstSample | uint64 | Absolute source sample offset, starting at zero |
| samples | uint32 | Valid samples, 0 through 320 |
| flags | uint32 | EOS=1, DISCONTINUITY=2, ERROR=4, AUTO_UTTERANCES=8, WAITING=16 |
| backlogSamples | uint32 | Valid queued samples after this block was removed |

Divide `firstSample` by 16000 for source-relative seconds. Header fields are
explicitly serialized; do not transmit a native C struct with padding. Unknown
flags, out-of-range unsigned values and sample-interval overflow are rejected.
ERROR requires EOS; a zero-sample block requires EOS or standalone WAITING.
Button capture uses WAITING while the recording controls are open; these blocks
do not advance the source position or count toward the utterance limit. A final zero-sample EOS
follows all valid samples, including a short final block. The server closes its
connection after EOS. The bridge caches terminal state for further target pulls.

## Firmware handshake

1. Initialization publishes `Regs[1]=magic` without connecting to the host.
2. Firmware writes `Regs[0]=1`; the callback blocks until one block is available.
3. Read `Regs[1..6]`: magic, firstSample low, firstSample high, samples, flags,
   backlogSamples. Check ERROR before using PCM.
4. Write `Regs[7]=0`, configure a 640-byte one-shot P2M DMA, and trigger it with
   the timer. `rdDataDMA(640)` consumes the staged PCM, pads it, and sets
   `Regs[7]=1` immediately before returning the payload to the model.
5. Poll register 7, then stop timer and DMA and consume only `samples` PCM values.

Register 8 exposes host monotonic milliseconds modulo 2^32, independent of FVP
simulation speed. KWS uses it for the two-second greeting timeout.

Further DMA reads without a new pull return zero padding, never replayed audio.
Network failures print an actionable host-start command and publish ERROR|EOS
with zero valid samples. Source failures propagate ERROR|EOS and print their
reason on the host. Restart both host and model for a new session.

## Python source API and tests

`AudioSource(options).pull()` starts acquisition lazily and returns an immutable
`AudioBlock(first_sample, pcm, flags, backlog_samples)`; its `samples` property
excludes padding. Call `close()` on disconnect or shutdown. `BlockQueue` provides
bounded storage independently of source acquisition.

```powershell
resources_downloaded/windows-executorch/env/Scripts/python.exe -m unittest discover -s tests/py -p "test_fvp_audio_demo*.py" -v
```

Tests generate WAV fixtures inside the repository and use mocked microphones.
Two transport tests require permission to open ephemeral localhost sockets.
