# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Create and validate an INT8-state PT2/PTE variant in the existing WSL environment."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import torch

from asr_zh_int8_state_pt2 import compare_clip, export_states
from asr_zh_int8_state_pte import export_host
from prepare_asr_zh import load_helpers
from validate_asr_zh_pte import validate_model


def write_header(path: Path, parameters: list[dict], digest: str):
    """Write the per-state real-zero encodings next to the matching model."""
    path.write_text(
        "// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates\n"
        "// <open-source-office@arm.com>\n// SPDX-License-Identifier: Apache-2.0\n"
        "#pragma once\n#include <cstdint>\nnamespace arm::app::asr_zh::state_quant {\n"
        'inline constexpr const char kPteSha256[] = "' + digest + '";\n'
        "inline constexpr int32_t kZeroPoints[73] = {\n" +
        ",\n".join("    " + ", ".join(str(row["zero_point"]) for row in parameters[i:i + 12])
                    for i in range(0, 73, 12)) + "\n};\n"
        "inline constexpr float kScales[73] = {\n" +
        ",\n".join("    " + ", ".join(f'{row["scale"]:.9e}f' for row in parameters[i:i + 4])
                    for i in range(0, 73, 4)) + "\n};\n}\n")


def validate_clips(source: Path, output: Path, helpers: object,
                   parameters: list[dict], clips: list[Path]) -> list[dict]:
    """Run only the requested short clips with independent evolving state histories."""
    old = torch.export.load(source / "zipformer_streaming_int8.pt2").module()
    new = torch.export.load(output / "zipformer_streaming_int8_state_io.pt2").module()
    tokenizer = helpers.sentencepiece.SentencePieceProcessor(
        model_file=str(source.parents[1] / "downloads/lang_bbpe_1000/bbpe.model"))
    results = []
    with torch.inference_mode():
        for clip in clips:
            result = compare_clip(old, new, list(helpers.audio.chunks(
                helpers.audio.features(clip), 77, 64)), parameters)
            tokens = [token for token, _ in itertools.groupby(sum(result["argmax_paths"], []))
                      if token != 0]
            result.update({"audio": str(clip),
                           "audio_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
                           "transcript": helpers.byte.smart_byte_decode(
                               tokenizer.Decode(tokens)).replace(" ", "")})
            results.append(result)
    return results


def publish_artifacts(source: Path, output: Path, name: str, result: dict):
    """Copy the separate model plus its matching reset metadata into MLEK resources."""
    manifest = json.loads((source / "manifest.json").read_text())
    manifest.update({"pte": name, "sha256": result["pte_sha256"], "state_io": "INT8",
                     "runtime_validation_status":
                         "PT2 equivalence and static PTE I/O PASS; FVP pending"})
    for row in manifest["inputs"][1:74]:
        row["dtype"] = "CHAR"
    if "outputs" in manifest:
        for row in manifest["outputs"][1:74]:
            row["dtype"] = "CHAR"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    destination = (Path(__file__).resolve().parents[3] /
                   "resources_downloaded/asr_zh/int8-state-768k")
    destination.mkdir(exist_ok=True)
    for filename in [name, "StateQuantization.hpp", "int8_state_report.json", "manifest.json"]:
        shutil.copy2(output / filename, destination / filename)


def main():
    """Export a separate variant; never overwrite the validated FP32-state baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zipformer-root", type=Path, required=True)
    parser.add_argument("--clip", type=Path, action="append", required=True)
    args = parser.parse_args()
    root = args.zipformer_root
    helpers = load_helpers(root)
    torch.set_num_threads(1)
    source = root / "artifacts/u85-1024-dedicated-768k-bindings-fixed"
    output = root / "artifacts/u85-1024-dedicated-768k-int8-state"
    output.mkdir(exist_ok=True)
    parameters = export_states(source / "zipformer_streaming_int8.pt2",
                               output / "zipformer_streaming_int8_state_io.pt2")
    print("PT2_INT8_STATE_EXPORTED", flush=True)
    name = "zipformer_small_ctc_zh_u85_1024_dedicated_768k_int8_state.pte"
    result = export_host(source / "zipformer_small_ctc_zh_u85_1024_dedicated_768k.pte",
                         output / name, parameters)
    result.update({"state_quantization": parameters,
                   "pt2_clips": validate_clips(source, output, helpers, parameters, args.clip),
                   "source_pt2_sha256": hashlib.sha256(
                       (source / "zipformer_streaming_int8.pt2").read_bytes()).hexdigest(),
                   "pt2_sha256": hashlib.sha256(
                       (output / "zipformer_streaming_int8_state_io.pt2").read_bytes()).hexdigest()
                   })
    validate_model(output / name)
    write_header(output / "StateQuantization.hpp", parameters, result["pte_sha256"])
    (output / "int8_state_report.json").write_text(json.dumps(result, ensure_ascii=False,
                                                            indent=2) + "\n")
    publish_artifacts(source, output, name, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ["state_quantization", "pt2_clips"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
