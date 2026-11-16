# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Recompile Zipformer Vela delegates for a partitioned Shared_Sram layout."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np

from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.ethosu.backend import EthosUBackend
from executorch.exir._serialize import _deserialize_pte_binary
from executorch.exir._serialize._program import serialize_pte_binary

from validate_asr_zh_pte import validate_model


def digest(data: bytes) -> str:
    """Return the SHA256 hex digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def vela_spec(root: Path, mode: str, config: Path, output: Path) -> EthosUCompileSpec:
    """Build the exact U85-1024 compiler settings for one SRAM mode.

    :param root:   Zipformer export project root.
    :param mode:   Vela memory mode.
    :param config: Vela system and memory placement configuration.
    :param output: Directory for Vela diagnostic artifacts.
    :returns:      Configured ExecuTorch Ethos-U compile specification.
    """
    output.mkdir(parents=True, exist_ok=True)
    flags = ["--output-format=raw", "--debug-force-regor", "--cop-format=COP1",
             "--optimise=Performance"]
    if mode == "Dedicated_Sram":
        flags.append("--arena-cache-size=786432")
    else:
        flags.append("--arena-cache-size=3145728")
    spec = EthosUCompileSpec(
        target="ethos-u85-1024",
        system_config="Ethos_U85_SYS_DRAM_Mid_1024",
        memory_mode=mode,
        config_ini=str(config),
        extra_flags=flags)
    spec.dump_intermediate_artifacts_to(str(output))
    return spec


def compile_tosa(path: Path, root: Path, mode: str, config: Path, output: Path) -> bytes:
    """Compile one serialized TOSA delegate and require a single raw Vela stream.

    :param path: TOSA FlatBuffer emitted by the validated PT2 graph.
    :param root:   Zipformer export project root.
    :param mode:   Vela memory mode.
    :param config: Vela system and memory placement configuration.
    :param output: Directory for Vela intermediate artifacts.
    :returns:    Raw COP1 command and weight stream.
    """
    spec = vela_spec(root, mode, config, output)
    result = EthosUBackend._compile_tosa_flatbuffer(path.read_bytes(), spec)
    assert not result.external_blocks, "Expected one inline Vela delegate stream"
    if mode == "Shared_Sram":
        with np.load(output / "output/out_vela.npz", allow_pickle=False) as compiled:
            assert int(compiled["scratch_shape"][0]) <= 3145728
            assert int(compiled["scratch_fast_shape"][0]) == 0
    return result.processed_bytes


def update_delegate_metadata(program: object, artifact_dir: Path, config: Path) -> None:
    """Mark every delegate with its standard Shared_Sram and 3 MiB arena settings."""
    plans = [plan for plan in program.execution_plan if plan.name == "forward"]
    assert len(plans) == 1
    for delegate in plans[0].delegates:
        assert delegate.id == "EthosUBackend"
        found_flags = False
        for compile_spec in delegate.compile_specs:
            if compile_spec.key == "compile_flags":
                tokens = compile_spec.value.decode("utf-8").split()
                tokens = [token for token in tokens
                          if not token.startswith("--arena-cache-size=")
                          and not token.startswith("--memory-mode=")
                          and not token.startswith("--config=")]
                tokens.extend(["--arena-cache-size=3145728", "--memory-mode=Shared_Sram",
                               f"--config={config}"])
                compile_spec.value = " ".join(tokens).encode("utf-8")
                found_flags = True
            elif compile_spec.key == "debug_artifact_path":
                compile_spec.value = str(artifact_dir).encode("utf-8")
        assert found_flags, "Ethos-U delegate compile flags are missing"


def build_shared_pte(root: Path, output: Path) -> dict:
    """Replace only the Vela streams and metadata in the validated INT8-state PTE.

    :param root:   Zipformer export project root.
    :param output: Destination directory for the shared variant.
    :returns:      Compiler hashes and static delegate interface validation.
    """
    source_dir = root / "artifacts/u85-1024-dedicated-768k-int8-state"
    source_pte = source_dir / "zipformer_small_ctc_zh_u85_1024_dedicated_768k_int8_state.pte"
    tosa_dir = root / "artifacts/u85-1024-dedicated-768k-full8/intermediates"
    tag0 = tosa_dir / "output_tag0_TOSA-1.0+INT+int16+int4+cf.tosa"
    tag1 = tosa_dir / "output_tag1_TOSA-1.0+INT+int16+int4+cf.tosa"
    output.mkdir(parents=True, exist_ok=True)
    output_name = "zipformer_small_ctc_zh_u85_1024_shared_int8_state.pte"
    source_bytes = source_pte.read_bytes()
    pte = _deserialize_pte_binary(source_bytes)
    original_streams = [entry.data for entry in pte.program.backend_delegate_data]
    original_hashes = [digest(stream) for stream in original_streams]
    assert len(original_hashes) == 2 and len(set(original_hashes)) == 2

    shared_config = output / "vela_shared.ini"
    config_text = (root / "configs/vela.ini").read_text()
    config_text, replacements = re.subn(
        r"(?ms)^\[Memory_Mode\.Shared_Sram\]\n.*?(?=^\[Memory_Mode\.Dedicated_Sram\])",
        "[Memory_Mode.Shared_Sram]\nconst_mem_area=Axi1\n"
        "arena_mem_area=Axi0\ncache_mem_area=Axi0\narena_cache_size=3145728\n\n",
        config_text)
    assert replacements == 1
    shared_config.write_text(config_text)

    direct_tag0 = (root / "artifacts/u85-1024-dedicated-768k-full8/direct_vela_tag0.bin").read_bytes()
    assert digest(direct_tag0) in original_hashes
    checked_tag1 = compile_tosa(
        tag1, root, "Dedicated_Sram", root / "configs/vela.ini",
        output / "vela-dedicated-check")
    assert digest(checked_tag1) in original_hashes
    assert {digest(direct_tag0), digest(checked_tag1)} == set(original_hashes)

    shared_by_original_hash = {
        digest(direct_tag0): compile_tosa(
            tag0, root, "Shared_Sram", shared_config, output / "vela-shared-tag0"),
        digest(checked_tag1): compile_tosa(
            tag1, root, "Shared_Sram", shared_config, output / "vela-shared-tag1"),
    }
    for delegate_index, stream in enumerate(original_streams):
        replacement = shared_by_original_hash[digest(stream)]
        pte.program.backend_delegate_data[delegate_index].data = replacement
    update_delegate_metadata(pte.program, output / "vela-shared-tag0", shared_config)

    pte_path = output / output_name
    pte_path.write_bytes(bytes(serialize_pte_binary(pte)))
    validation = validate_model(pte_path)
    new_hashes = [digest(entry.data) for entry in pte.program.backend_delegate_data]
    main_index = original_hashes.index(digest(direct_tag0))
    assert new_hashes[main_index] != original_hashes[main_index]
    pte_hash = digest(pte_path.read_bytes())
    header = (source_dir / "StateQuantization.hpp").read_text()
    header, substitutions = re.subn(
        r'kPteSha256\[\] = "[0-9a-f]{64}"',
        f'kPteSha256[] = "{pte_hash}"', header)
    assert substitutions == 1
    (output / "StateQuantization.hpp").write_text(header)
    return {"source_pte_sha256": digest(source_bytes),
            "pte_sha256": pte_hash,
            "memory_mode": "Shared_Sram",
            "system_config": "Ethos_U85_SYS_DRAM_Mid_1024",
            "vela_sram_target_bytes": 3145728,
            "vela_config": str(shared_config),
            "delegate_stream_sha256": new_hashes,
            "source_delegate_stream_sha256": original_hashes,
            "delegate_streams_recompiled": True,
            "delegate_streams_changed": sum(
                new != old for new, old in zip(new_hashes, original_hashes)),
            "vela_tosa_sources": [str(tag0), str(tag1)],
            "static_validation": validation}


def publish(output: Path, result: dict) -> None:
    """Copy the PTE and its matching INT8-state metadata into MLEK resources."""
    output_name = "zipformer_small_ctc_zh_u85_1024_shared_int8_state.pte"
    source_dir = output.parent / "u85-1024-dedicated-768k-int8-state"
    manifest = json.loads((source_dir / "manifest.json").read_text())
    manifest.update({"pte": output_name, "sha256": result["pte_sha256"],
                     "memory_mode": "Shared_Sram", "cache_bytes": 0,
                     "vela_sram_target_bytes": result["vela_sram_target_bytes"],
                     "vela_config": result["vela_config"],
                     "delegate_stream_sha256": result["delegate_stream_sha256"],
                     "delegate_streams_unchanged": False,
                     "delegate_streams_recompiled": True,
                     "delegate_streams_changed": result["delegate_streams_changed"],
                     "runtime_validation_status":
                         "PT2 INT8-state equivalence and static Shared_Sram PTE I/O PASS; FVP pending"})
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report_path = output / "shared_vela_report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n")

    repo_root = Path(__file__).resolve().parents[3]
    destination = repo_root / "resources_downloaded/asr_zh/int8-state-shared"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in [output_name, "StateQuantization.hpp", "manifest.json",
                     "shared_vela_report.json", "vela_shared.ini"]:
        source = output / filename
        shutil.copy2(source, destination / filename)


def main():
    """Compile Shared_Sram delegates and publish a separate PTE variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zipformer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.zipformer_root.resolve()
    output = args.output or root / "artifacts/u85-1024-shared-int8-state"
    result = build_shared_pte(root, output)
    publish(output, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "static_validation"}, indent=2), flush=True)
    print(json.dumps({"static_validation": result["static_validation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
