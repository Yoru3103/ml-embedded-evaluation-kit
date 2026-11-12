# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Compare seeds and Euler schedules using host float and calibrated quantized graphs."""
import argparse
import json
import os
from pathlib import Path

from mlek_tools.pico_face.checkpoints import PICO_FACES_FILES


def main():
    """Generate labeled host references; these are not FPGA measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--vela-config", type=Path, required=True)
    parser.add_argument("--variant", choices=PICO_FACES_FILES, default="m3_long_cfg")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 7, 11])
    parser.add_argument("--steps", type=int, nargs="+", choices=[1, 2, 4, 8], default=[4, 8])
    args = parser.parse_args()
    if args.output is None:
        name = "comparison" if args.variant == "m3_long_cfg" else f"comparison-{args.variant}"
        args.output = Path("build/logs/pico_face_logs/out") / name
    os.environ["PICO_FACES_CHECKPOINTS"] = str(args.checkpoints.resolve())
    os.environ["PICO_FACES_VARIANT"] = args.variant
    os.environ["PICO_FACES_QUANT"] = "a16w8"
    os.environ["PICO_FACES_DECODE_QUANT"] = "a8w8"
    os.environ["PICO_FACES_CALIB_SEEDS"] = "16"
    import torch
    from PIL import Image, ImageDraw
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from mlek_tools.pico_face import model
    from mlek_tools.pico_face.export_support import quantize_method

    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    spec = EthosUCompileSpec(
        target="ethos-u85-1024", config_ini=str(args.vela_config.resolve()),
        system_config="Ethos_U85_SYS_DRAM_Mid_1024", memory_mode="Dedicated_Sram",
        extra_flags=["--arena-cache-size=393216"])
    methods = model.get_methods()
    quantized = {method.name: quantize_method(method, spec) for method in methods}
    condition = model.Cond.from_checkpoint().cond_table()
    records = []
    sheet = Image.new("RGB", (len(args.steps) * 276, len(args.seeds) * 290), "white")
    draw = ImageDraw.Draw(sheet)
    with torch.no_grad():
        for row, seed in enumerate(args.seeds):
            for column, steps in enumerate(args.steps):
                latent = model.noise(seed, 8, 16)
                latent = model.euler_sample(quantized["dit_step"], condition, latent,
                                            y=1, w=4.0, k_steps=steps)
                pixels = model.to_image(quantized["decode"](latent))
                image = Image.fromarray(pixels)
                name = f"host_fakequant_seed{seed}_steps{steps}"
                image.save(args.output / f"{name}.png")
                (args.output / f"{name}.rgb").write_bytes(pixels.tobytes())
                records.append({"source": "host_fakequant", "seed": seed, "steps": steps,
                                "class_id": 1, "guidance": 4.0, "image": f"{name}.png"})
                sheet.paste(image.resize((256, 256), Image.Resampling.NEAREST),
                            (column * 276 + 10, row * 290 + 25))
                draw.text((column * 276 + 10, row * 290 + 6),
                          f"Host fakequant | seed {seed} | {steps} steps", fill="black")
                print(f"Saved {name}", flush=True)
        float_sheet = Image.new("RGB", (len(args.steps) * 276, 580), "white")
        float_draw = ImageDraw.Draw(float_sheet)
        for column, steps in enumerate(args.steps):
            latent = model.euler_sample(methods[0].module, condition, model.noise(3, 8, 16),
                                        y=1, w=4.0, k_steps=steps)
            pixels = model.to_image(methods[1].module(latent))
            image = Image.fromarray(pixels)
            image.save(args.output / f"host_float_seed3_steps{steps}.png")
            for row, (label, reference) in enumerate([
                    ("float", image),
                    ("fakequant", Image.open(
                        args.output / f"host_fakequant_seed3_steps{steps}.png"))]):
                float_sheet.paste(reference.resize((256, 256), Image.Resampling.NEAREST),
                                  (column * 276 + 10, row * 290 + 25))
                float_draw.text((column * 276 + 10, row * 290 + 6),
                                f"Host {label} | seed 3 | {steps} steps", fill="black")
        sheet.save(args.output / "seed_steps_comparison.png")
        float_sheet.save(args.output / "seed3_float_quantized_comparison.png")
    (args.output / "comparison.json").write_text(
        json.dumps({"note": "Host references, not FPGA measurements or NPU execution",
                    "variant": args.variant, "calibration_seeds": 16, "images": records}, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
