# Chinese streaming Zipformer resources

Model: csukuangfj/icefall-streaming-zipformer-small-ctc-zh-2025-04-01.
Pinned Hugging Face revision: 6768aa8b65bc68e128af2ee08a7142dd0abe4be4.
The upstream model card declares Apache-2.0.

- `samples/demo.wav` is the upstream reference/test_wavs/0.wav, written as 16 kHz PCM16.
- `TokenBytes.hpp` is generated from the matching lang_bbpe_1000/bbpe.model and
  Icefall byte mapping. It includes blank/control IDs; it is not an English label list.
- `Validation.hpp` contains three fixed feature inputs plus per-chunk PT2 argmax paths
  and state samples for the demo.
- `validation.json` records source hashes, sample length, and PT2 reference results.
- Local FP32-state regression artifacts use the ignored
  `resources_downloaded/asr_zh/dedicated-768k/` directory.
- Native Fbank fixtures are deterministic synthetic PCM, generated with seed 20261028;
  they require no external dataset files.

Download the [INT8-state demo inputs](https://github.com/Yoru3103/ml-embedded-evaluation-kit/releases/download/model-resourse/asr-zh-zipformer-u85-dedicated-768k-int8-inputs.zip)
for U85-1024, Dedicated SRAM 768 KiB and COP1. The
[Chinese ASR demo guide](../../docs/sections/demos/asr_zh.md) shows where to place
the PTE and its state quantization metadata.
