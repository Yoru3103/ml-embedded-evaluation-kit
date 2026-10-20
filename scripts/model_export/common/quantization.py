# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Build compile specifications and calibrate through one PT2E implementation."""

import argparse
from pathlib import Path

import torch
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from model_export.common.images import load_image
from model_export.common.layout import model_input

def make_compile_spec(args: argparse.Namespace) -> EthosUCompileSpec:
    """Build the NPU compile specification without mutating argument lists."""
    flags = list(args.extra_flag)
    if args.arena_cache_size is not None:
        flags.append(f"--arena-cache-size={args.arena_cache_size}")
    return EthosUCompileSpec(target=args.target, config_ini=str(args.vela_config.resolve()),
                            system_config=args.system_config, memory_mode=args.memory_mode,
                            extra_flags=flags)


def quantize(model: torch.nn.Module, paths: list[Path], args: argparse.Namespace,
             spec: EthosUCompileSpec) -> torch.fx.GraphModule:
    """Export and calibrate using representative images without tracking gradients."""
    # 导入第一张校准图片
    example = load_image(paths[0], args.image_size, args.preprocessing,
                         getattr(args, "input_channels", 3))
    layout = getattr(args, "io_layout", "nchw")
    example = model_input(example, layout)
    # 将浮点模型捕获为计算图
    exported = torch.export.export(model, (example,), strict=True)
    graph_dir = args.output.parent / f"{args.output.stem}_graphs"
    if args.save_intermediate:
        graph_dir.mkdir(parents=True, exist_ok=True)
        torch.export.save(exported, graph_dir / "float.pt2")    # 可选，供Netron查看量化前的计算图
    # 指定Ethos-U支持的量化规则，并为全图设置默认规则。--per-channel 控制权重是否按输出通道分别量化。
    quantizer = EthosUQuantizer(spec).set_global(
        get_symmetric_quantization_config(is_per_channel=args.per_channel))
    # 按规则在计算图中加入观察器，得到等待校准的prepared模型
    prepared = prepare_pt2e(exported.module(check_guards=False), quantizer)
    with torch.no_grad():
        for index, path in enumerate(paths, 1):
            prepared(model_input(load_image(path, args.image_size, args.preprocessing,
                                             getattr(args, "input_channels", 3)), layout))
            if index % 16 == 0 or index == len(paths):
                print(f"Calibrated {index}/{len(paths)}", flush=True)
    # 根据统计结果计算量化参数，生成量化后的计算图
    quantized = convert_pt2e(prepared)
    if args.save_intermediate:
        torch.export.save(torch.export.export(quantized, (example,), strict=True),
                          graph_dir / "quantized.pt2")
        print(f"Netron intermediate models: {graph_dir}", flush=True)
    return quantized
