# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned Zipformer assets and native/FVP fixtures from a WSL export project."""
import argparse
import hashlib
from importlib import import_module, metadata
import itertools
import json
from pathlib import Path
import shutil
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

HEADER = (
    "// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates\n"
    "// <open-source-office@arm.com>\n"
    "// SPDX-License-Identifier: Apache-2.0\n#pragma once\n"
)


def sha256(path: Path) -> str:
    """Calculate a local asset fingerprint.

    :param path:    Input file.
    :returns:       SHA256 hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floats(values: np.ndarray) -> str:
    """Format C++ float literals.

    :param values:  Values in C order.
    :returns:       Comma-separated initializer.
    """
    return ", ".join(f"{float(value):.9e}f" for value in values.flatten())


def load_helpers(source: Path) -> SimpleNamespace:
    """Load helpers from the explicitly requested export project."""
    sys.path[:0] = [str(source / ".deps"), str(source / "scripts"),
                    str(source / "src/upstream")]
    import_module("torchao.quantization.pt2e.quantize_pt2e")
    import_module("executorch.exir")
    return SimpleNamespace(audio=import_module("audio_utils"),
                           byte=import_module("byte_utils"),
                           fbank=import_module("kaldi_native_fbank"),
                           sentencepiece=import_module("sentencepiece"))


def write_cpp(path: Path, lines: list[str]):
    """Wrap generated initializers without breaking tokens or escape sequences."""
    wrapped = []
    for line in "\n".join(lines).splitlines():
        wrapped.extend(textwrap.wrap(line, width=100, subsequent_indent="    ",
                                     break_long_words=False, break_on_hyphens=False) or [""])
    path.write_text("\n".join(wrapped) + "\n")


def prepare_files(repo: Path, source: Path, artifact: Path | None = None) -> dict:
    """Copy the verified PTE and a PCM16 reference clip into the repository."""
    paths = {
        "source": source, "resources": repo / "resources/asr_zh",
        "fixtures": repo / "tests/processing/asr_zh",
        "models": repo / "resources_downloaded/asr_zh/dedicated-768k",
        "artifact": artifact or source / "artifacts/u85-1024-dedicated-768k",
    }
    for name in ["resources", "fixtures", "models"]:
        paths[name].mkdir(parents=True, exist_ok=True)
    (paths["resources"] / "samples").mkdir(exist_ok=True)
    paths["manifest"] = json.loads((paths["artifact"] / "manifest.json").read_text())
    paths["pte"] = paths["artifact"] / paths["manifest"]["pte"]
    assert sha256(paths["pte"]) == paths["manifest"]["sha256"]
    validation = import_module("validate_asr_zh_pte").validate_model(paths["pte"])
    (paths["models"] / "delegate_io.json").write_text(json.dumps(validation, indent=2) + "\n")
    shutil.copy2(paths["pte"], paths["models"] / paths["pte"].name)
    shutil.copy2(paths["artifact"] / "manifest.json", paths["models"] / "export_manifest.json")
    paths["wav"] = paths["resources"] / "samples/demo.wav"
    pcm, rate = sf.read(source / "downloads/reference/test_wavs/0.wav", dtype="int16")
    assert rate == 16000 and pcm.ndim == 1
    sf.write(paths["wav"], pcm, rate, subtype="PCM_16")
    paths["samples"] = len(pcm)
    return paths


def write_tokens(resources: Path, source: Path, helpers: SimpleNamespace) -> object:
    """Generate exact ID-to-byte entries; return the matching SentencePiece processor."""
    tokenizer = helpers.sentencepiece.SentencePieceProcessor(
        model_file=str(source / "downloads/lang_bbpe_1000/bbpe.model"))
    table = [HEADER, '#include "mlek/use_case/asr_zh/CtcDecoder.hpp"',
             "namespace arm::app::asr_zh::assets {"]
    names = []
    for index in range(tokenizer.GetPieceSize()):
        if tokenizer.IsControl(index):
            data = b""
        elif tokenizer.IsUnknown(index):
            data = b" "
        else:
            piece = tokenizer.IdToPiece(index).replace("\u2581", " ")
            data = bytes(helpers.byte.BCHAR_TO_BYTE[char] for char in piece)
        table.append(f"inline const unsigned char kToken{index}[] = {{" +
                     ", ".join(str(byte) for byte in data + b"\0") + "};")
        names.append(f"{{kToken{index}, {len(data)}}}")
    table.append("inline const TokenBytes kTokens[] = {" + ", ".join(names) + "};")
    table.append("} // namespace arm::app::asr_zh::assets")
    write_cpp(resources / "TokenBytes.hpp", table)
    return tokenizer


def write_fbank_fixture(fixtures: Path, module: object):
    """Generate non-speech fixtures so native tests need no external audio files."""
    rng = np.random.default_rng(20261028)
    synthetic = rng.integers(-12000, 12000, size=21120, dtype=np.int16)
    options = module.FbankOptions()
    options.frame_opts.dither = 0
    options.frame_opts.snip_edges = False
    options.mel_opts.num_bins = 80
    options.mel_opts.low_freq = 20
    options.mel_opts.high_freq = -400
    fbank = module.OnlineFbank(options)
    fbank.accept_waveform(16000, (synthetic.astype(np.float32) / 32768).tolist())
    fbank.input_finished()
    expected = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
    chosen = [0, 1, 63, 64, 76, 77, len(expected) - 1]
    fixture = [HEADER, "#include <cstdint>\n#include <cstddef>",
               "namespace asr_zh_fixture {", "inline const int16_t kPcm[] = {" +
               ", ".join(str(value) for value in synthetic) + "};",
               "inline const size_t kFrames[] = {" + ", ".join(map(str, chosen)) + "};",
               "inline const float kExpected[][80] = {" +
               ",\n".join("{" + floats(expected[i]) + "}" for i in chosen) + "};", "}"]
    write_cpp(fixtures / "FbankFixture.hpp", fixture)


def run_reference(paths: dict, helpers: SimpleNamespace) -> dict:
    """Run one INT8 PT2 reference utterance with its own evolving state."""
    torch.set_num_threads(1)
    qpath = paths["artifact"] / "zipformer_streaming_int8.pt2"
    model = torch.export.load(qpath).module()
    states = tuple(torch.zeros(item["shape"], dtype=torch.int32 if item["dtype"] == "INT"
                               else torch.float32) for item in paths["manifest"]["inputs"][1:])
    data = {"path": [], "state_samples": [], "int8_pt2_sha256": sha256(qpath)}
    reference = {}
    data["features"] = list(helpers.audio.chunks(helpers.audio.features(paths["wav"]), 77, 64))
    with torch.inference_mode():
        for chunk, features in enumerate(data["features"]):
            output = model(features, *states)
            if chunk < 3:
                reference.update({f"chunk{chunk}_output{slot}":
                                  value.to(torch.int32).numpy() if slot == 74 else value.numpy()
                                  for slot, value in enumerate(output)})
            states = tuple(value.to(template.dtype) for value, template in zip(output[1:], states))
            data["path"].append(output[0].argmax(-1).flatten().tolist())
            data["state_samples"].append([[float(value.flatten()[i]) for i in
                                           [0, value.numel() // 2, value.numel() - 1]]
                                          for value in states])
    np.savez_compressed(paths["models"] / "pt2_reference.npz", **reference)
    return data


def write_validation(resources: Path, data: dict, tokenizer: object,
                     helpers: SimpleNamespace) -> str:
    """Write three fixed inputs and path/state samples for the complete demo."""
    collapsed = [token for token, _ in itertools.groupby(sum(data["path"], [])) if token != 0]
    text = helpers.byte.smart_byte_decode(tokenizer.Decode(collapsed)).replace(" ", "")
    encoded = text.encode("utf-8")
    golden = [HEADER, "namespace arm::app::asr_zh::assets {",
              "inline const float kFixedFeatures[3][77 * 80] = {" +
              ",\n".join("{" + floats(chunk.numpy()) + "}"
                         for chunk in data["features"][:3]) + "};",
              "inline const unsigned kExpectedPath[][16] = {" +
              ",\n".join("{" + ", ".join(map(str, row)) + "}" for row in data["path"]) + "};",
              "inline const float kExpectedStates[][74][3] = {" +
              ",\n".join("{" + ", ".join("{" + floats(np.array(value)) + "}" for value in row)
                         + "}" for row in data["state_samples"]) + "};",
              "inline const char kExpectedText[] = " +
              "\n".join('"' + "".join(f"\\x{byte:02x}" for byte in encoded[i:i + 16]) + '"'
                        for i in range(0, len(encoded), 16)) + ";", "}"]
    write_cpp(resources / "Validation.hpp", golden)
    return text


def main():
    """Generate resources without training, calibration, export or dataset evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zipformer-root", required=True, type=Path)
    parser.add_argument("--artifact-directory", type=Path)
    args = parser.parse_args()
    source = args.zipformer_root.resolve()
    helpers = load_helpers(source)
    paths = prepare_files(Path(__file__).resolve().parents[3], source, args.artifact_directory)
    tokenizer = write_tokens(paths["resources"], source, helpers)
    write_fbank_fixture(paths["fixtures"], helpers.fbank)
    data = run_reference(paths, helpers)
    data["expected_text"] = write_validation(paths["resources"], data, tokenizer, helpers)
    data.update({
        "pte_sha256": sha256(paths["pte"]),
        "tokenizer_sha256": sha256(source / "downloads/lang_bbpe_1000/bbpe.model"),
        "demo_wav_sha256": sha256(paths["wav"]),
        "sample_source": "Zipformer official reference/test_wavs/0.wav",
        "sample_rate": 16000, "samples": paths["samples"], "chunks": len(data["features"]),
        "knf_version": metadata.version("kaldi-native-fbank"),
    })
    del data["features"]
    (paths["resources"] / "validation.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    keys = ["expected_text", "samples", "chunks", "pte_sha256"]
    print(json.dumps({key: data[key] for key in keys},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
