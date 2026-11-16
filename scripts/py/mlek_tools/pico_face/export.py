# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Export matching pico-face PTE, conditioning table and provenance for MLEK."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import struct
from pathlib import Path

from mlek_tools.pico_face.checkpoints import PICO_FACES_FILES


def scratch_sizes(data: bytes) -> dict[str, int]:
    """Read per-method scratch sizes from the serialized Vela blocks."""
    from executorch.exir._serialize import _deserialize_pte_binary

    program = _deserialize_pte_binary(data).program
    sizes = {}
    for plan in program.execution_plan:
        for delegate in plan.delegates:
            blob = bytes(program.backend_delegate_data[delegate.processed.index].data)
            offset = 0
            while offset + 32 <= len(blob):
                name = blob[offset:offset + 16].split(b"\0", 1)[0]
                length = struct.unpack_from("<I", blob, offset + 16)[0]
                offset += 32
                if offset + length > len(blob):
                    raise ValueError("Truncated Vela block")
                if name == b"scratch_size":
                    sizes[plan.name] = struct.unpack_from("<I", blob, offset)[0]
                    break
                offset += (length + 15) // 16 * 16
    if set(sizes) != {"dit_step", "decode"}:
        raise ValueError("Expected scratch metadata for both methods")
    return sizes


def main():
    """Export U85 Z1024 with the requested SRAM memory mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True,
                        help="Parent directory of the model variant checkpoint directory")
    parser.add_argument("--variant", choices=PICO_FACES_FILES, default="m3_long_cfg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vela-config", type=Path, required=True)
    parser.add_argument("--memory-mode", choices=("Dedicated_Sram", "Shared_Sram"),
                        default="Dedicated_Sram")
    args = parser.parse_args()
    os.environ["PICO_FACES_CHECKPOINTS"] = str(args.checkpoints.resolve())
    os.environ["PICO_FACES_VARIANT"] = args.variant
    os.environ["PICO_FACES_QUANT"] = "a16w8"
    os.environ["PICO_FACES_DECODE_QUANT"] = "a8w8"
    os.environ["PICO_FACES_CALIB_SEEDS"] = "16"
    os.environ["AI_LAYER_STRICT"] = "1"
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from mlek_tools.pico_face import model
    from mlek_tools.pico_face.export_support import export_model, params_header

    spec = EthosUCompileSpec(
        target="ethos-u85-1024", config_ini=str(args.vela_config.resolve()),
        system_config="Ethos_U85_SYS_DRAM_Mid_1024", memory_mode=args.memory_mode,
        extra_flags=["--arena-cache-size=393216"]
        if args.memory_mode == "Dedicated_Sram" else [])
    args.output.mkdir(parents=True, exist_ok=True)
    data, cpu_ops = export_model(spec, model)
    scratch = scratch_sizes(data)
    print(f"Per-method NPU scratch bytes: {scratch}")
    params = params_header(model.get_params())
    (args.output / "pico_face.pte").write_bytes(data)
    (args.output / "PicoFaceParams.hpp").write_text(params, encoding="utf-8", newline="\n")
    manifest = {
        "upstream_commit": "0d600669b83f2daf49e1a30b9ec37027c6446030",
        "variant": model.VARIANT, "quantization": model.QUANTIZATION,
        "decode_quantization": model.DECODE_QUANTIZATION,
        "calibration_seeds": model.CALIB_SEEDS,
        "target": "ethos-u85-1024", "memory_mode": args.memory_mode,
        "system_config": "Ethos_U85_SYS_DRAM_Mid_1024",
        "cache_bytes": 393216 if args.memory_mode == "Dedicated_Sram" else 0,
        "scratch_bytes": scratch,
        "pte_sha256": hashlib.sha256(data).hexdigest(),
        "params_sha256": hashlib.sha256(params.encode()).hexdigest(),
        "cpu_ops": sorted(cpu_ops),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "executorch", "torchao", "ethos-u-vela")},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(data)} bytes to {args.output}")


if __name__ == "__main__":
    main()
