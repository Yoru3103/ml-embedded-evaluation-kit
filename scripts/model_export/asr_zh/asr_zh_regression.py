# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Prepare a small AISHELL boundary suite and compare FVP with evolving/teacher-forced PT2."""
import argparse
import itertools
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf
import torch

from prepare_asr_zh import load_helpers


def initial_states(parameters: list) -> tuple:
    """Encode real zero independently for each INT8 streaming state."""
    return tuple(torch.full(row["shape"], row["zero_point"], dtype=torch.int8)
                 for row in parameters) + (torch.zeros(1, dtype=torch.int32),)


def load_model(root: Path) -> tuple:
    """Load the existing calibrated INT8-state graph and reset parameters."""
    folder = root / "artifacts/u85-1024-dedicated-768k-int8-state"
    parameters = json.loads((folder / "int8_state_report.json").read_text())["state_quantization"]
    model = torch.export.load(folder / "zipformer_streaming_int8_state_io.pt2").module()
    return model, parameters


def make_cases(root: Path) -> tuple:
    """Select two test utterances and their deterministic boundary variants."""
    data_root = root.parent / "conformer_zh/data"
    entries = [json.loads(line) for line in
               (data_root / "manifests/test.jsonl").read_text().splitlines()]
    chosen = [next(row for row in entries if row["id"].endswith(name))
              for name in ["BAC009S0913W0279", "BAC009S0904W0429"]]
    audio = [sf.read(data_root / "aishell" / row["wav"], dtype="int16")[0]
             for row in chosen]
    cases = [("00_utterance_a", audio[0], chosen[0]["text"]),
             ("01_utterance_b", audio[1], chosen[1]["text"]),
             ("02_crop_1370ms", audio[0][8000:8000 + 21920], None),
             ("03_crop_410ms", audio[1][16000:16000 + 6560], None),
             ("04_silence_1280ms", np.zeros(20480, dtype=np.int16), ""),
             ("05_repeat_a", audio[0], chosen[0]["text"]),
             ("06_concat_ab", np.concatenate(audio), None)]
    return chosen, cases


def reference_clip(model: object, parameters: list, helpers: object,
                   path: Path, tokenizer: object) -> tuple:
    """Run one clip with its own evolving state and retain three output sets."""
    states, paths, tensors = initial_states(parameters), [], {}
    with torch.inference_mode():
        for chunk, features in enumerate(
                helpers.audio.chunks(helpers.audio.features(path), 77, 64)):
            values = model(features, *states)
            paths.append(values[0].argmax(-1).flatten().tolist())
            if chunk < 3:
                for slot, value in enumerate(values):
                    tensors[f"b{chunk}_o{slot}"] = value.numpy()
            states = tuple(values[1:])
    ids = [token for token, _ in itertools.groupby(sum(paths, [])) if token != 0]
    text = helpers.byte.smart_byte_decode(tokenizer.Decode(ids)).replace(" ", "")
    return {"chunks": len(paths), "pt2_text": text, "paths": paths}, tensors


def prepare(root: Path, output: Path):
    """Create seven local AISHELL clips and PT2 references without corpus evaluation."""
    helpers = load_helpers(root)
    model, parameters = load_model(root)
    chosen, cases = make_cases(root)
    (output / "audio").mkdir(parents=True, exist_ok=True)
    tokenizer = helpers.sentencepiece.SentencePieceProcessor(
        model_file=str(root / "downloads/lang_bbpe_1000/bbpe.model"))
    records, tensors = [], {}
    for clip, case in enumerate(cases):
        path = output / "audio" / f"{case[0]}.wav"
        sf.write(path, case[1], 16000, subtype="PCM_16")
        record, values = reference_clip(model, parameters, helpers, path, tokenizer)
        record.update(file=path.name, samples=len(case[1]), dataset_reference=case[2])
        records.append(record)
        tensors.update({f"c{clip}_{key}": value for key, value in values.items()})
        print(case[0], record["chunks"], record["pt2_text"], flush=True)
    np.savez_compressed(output / "references.npz", **tensors)
    (output / "cases.json").write_text(json.dumps(
        {"dataset": "AISHELL-1 test", "sources": chosen, "cases": records},
        ensure_ascii=False, indent=2) + "\n")


def read_dump(path: Path, templates: list) -> list:
    """Read exactly one concatenated tensor set using its logical dtypes and shapes."""
    data, offset, result = path.read_bytes(), 0, []
    for template in templates:
        value = np.frombuffer(data, dtype=template.dtype, count=template.size,
                              offset=offset).reshape(template.shape).copy()
        result.append(value)
        offset += template.nbytes
    assert offset == len(data), (path, offset, len(data))
    return result


def compare_state_sets(model: object, parameters: list, values: tuple,
                       position: tuple) -> list:
    """Separate single-step backend error from independently evolving state error."""
    expected, inputs, actual = values
    teacher = model(*(torch.from_numpy(value) for value in inputs))
    differences = []
    for slot in range(75):
        scale = parameters[slot - 1]["scale"] if 1 <= slot <= 73 else 1
        own = np.abs(actual[slot].astype(np.float64) - expected[slot]) * scale
        forced = np.abs(actual[slot].astype(np.float64) - teacher[slot].numpy()) * scale
        differences.append({"clip": position[0], "chunk": position[1], "slot": slot,
                            "own_state_mae": float(own.mean()),
                            "same_input_mae": float(forced.mean()),
                            "same_input_max": float(forced.max())})
        if slot == 74:
            assert np.array_equal(actual[slot], expected[slot])
    return differences


def check_state_input(inputs: list, parameters: list, build: Path,
                      templates: list, position: tuple):
    """Require exact per-clip reset bytes and exact preceding-output feedback."""
    if position[1] == 0:
        expected = [state.numpy() for state in initial_states(parameters)]
    else:
        expected = read_dump(
            build / f"asr_zh_{position[0]:03}_{position[1] - 1:03}_outputs.bin", templates)[1:]
    assert all(np.array_equal(actual, target) for actual, target in zip(inputs[1:], expected)), (
        "State reset/feedback mismatch", position)


def compare_dumps(root: Path, output: Path, build: Path, cases: list) -> list:
    """Compare the first three complete tensor dumps per clip with PT2."""
    model, parameters = load_model(root)
    differences = []
    with np.load(output / "references.npz") as reference, torch.inference_mode():
        for clip, case in enumerate(cases):
            for chunk in range(min(3, case["chunks"])):
                expected = [reference[f"c{clip}_b{chunk}_o{slot}"] for slot in range(75)]
                input_templates = [np.zeros((1, 77, 80), np.float32)] + [
                    state.numpy() for state in initial_states(parameters)]
                inputs = read_dump(build / f"asr_zh_{clip:03}_{chunk:03}_inputs.bin",
                                   input_templates)
                actual = read_dump(build / f"asr_zh_{clip:03}_{chunk:03}_outputs.bin", expected)
                check_state_input(inputs, parameters, build, expected, (clip, chunk))
                differences.extend(compare_state_sets(
                    model, parameters, (expected, inputs, actual), (clip, chunk)))
    return differences


def compare(root: Path, output: Path, build: Path, log: Path):
    """Check all texts/paths and quantify state errors under identical FVP/PT2 inputs."""
    load_helpers(root)
    cases = json.loads((output / "cases.json").read_text())["cases"]
    text = log.read_text(errors="replace")
    assert "ASR_ZH completed success=1" in text and "ASR_ZH_FAILED" not in text
    texts = re.findall(r"ASR_ZH_TEXT clip=(\d+) chunks=(\d+) text=([^\r\n]*)", text)
    assert len(texts) == len(cases)
    rows = []
    for clip, case in enumerate(cases):
        entry = texts[clip]
        assert int(entry[1]) == case["chunks"]
        paths = [list(map(int, row.split())) for row in re.findall(
            rf"ASR_ZH_REGRESSION clip={clip} chunk=\d+ path=([0-9 ]+)", text)]
        assert len(paths) == case["chunks"]
        agreement = np.mean(np.array(paths) == np.array(case["paths"]))
        rows.append({"file": case["file"], "chunks": int(entry[1]),
                     "pt2_text": case["pt2_text"], "fvp_text": entry[2],
                     "text_match": entry[2] == case["pt2_text"],
                     "argmax_agreement": float(agreement)})
    assert rows[0]["fvp_text"] == rows[5]["fvp_text"], "State reset repeat differs"
    assert all((build / f"asr_zh_000_{chunk:03}_outputs.bin").read_bytes() ==
               (build / f"asr_zh_005_{chunk:03}_outputs.bin").read_bytes() for chunk in range(3)), (
        "Repeated clip output bytes differ")
    report = {"cases": rows, "all_texts_match_pt2": all(row["text_match"] for row in rows),
              "repeat_reset_pass": True,
              "state_differences": compare_dumps(root, output, build, cases),
              "scope": "Seven local boundary clips; no full evaluation or hardware timing."}
    (output / "fvp_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "state_differences"},
                     ensure_ascii=False, indent=2))


def main():
    """Prepare the small suite or compare a completed multi-clip FVP run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "compare"])
    parser.add_argument("--zipformer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "prepare":
        prepare(args.zipformer_root, args.output)
    else:
        assert args.build and args.log
        compare(args.zipformer_root, args.output, args.build, args.log)


if __name__ == "__main__":
    main()
