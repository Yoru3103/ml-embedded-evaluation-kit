#  SPDX-FileCopyrightText:  Copyright 2021, 2023, 2026 Arm Limited and/or
#  its affiliates <open-source-office@arm.com>
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Utility script to generate model c file that can be included in the
project directly. This should be called as part of CMake framework
should the models need to be generated at configuration stage.

This script is intentionally thin: it delegates validation to
format-specific modules and focuses on producing the .cc output.
"""
import binascii
from argparse import ArgumentParser
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from mlek_tools.gen.gen_utils import gen_header
from mlek_tools.validation.npu_validation import NpuValidationArgs
from mlek_tools.validation.pte_validator import validate_pte_model
from mlek_tools.validation.tflite_validator import validate_tflite_model


def get_model_data(model_path: str) -> list[str]:
    """Read a binary model and format its C array without growing a large string.

    :param model_path:  Path to the binary model.
    :returns:           One C array initializer, retaining the existing byte/line format.
    """
    # Read raw bytes so we can emit a C array initializer.
    with open(model_path, 'rb') as model_file:
        data = model_file.read()

    bytes_per_line = 32
    lines = []
    for offset in range(0, len(data), bytes_per_line):
        hexstream = binascii.hexlify(data[offset:offset + bytes_per_line]).decode('ascii')
        lines.append("\n" + "".join(
            '0x' + hexstream[index:index + 2] + ", "
            for index in range(0, len(hexstream), 2)))
    return ['{' + ''.join(lines) + '};\n']


def main():
    """
    Generate models .cpp
    """
    # pylint: disable=duplicate-code
    parser = ArgumentParser()

    parser.add_argument(
        "--model_path",
        help="Model path (.tflite or .pte)",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        help="Output directory",
        required=True
    )

    parser.add_argument(
        '-e',
        '--expression',
        action='append',
        default=[],
        dest="expr"
    )

    parser.add_argument(
        '--header',
        action='append',
        default=[],
        dest="headers"
    )

    parser.add_argument(
        '-ns',
        '--namespaces',
        action='append',
        default=[],
        dest="namespaces"
    )

    parser.add_argument(
        "--license_template",
        type=str,
        help="Header template file",
        default="header_template.txt"
    )

    parser.add_argument(
        "--ethos_u_memory_mode",
        type=str,
        help="Arm Ethos-U NPU Vela memory mode (Shared_Sram, Sram_Only, Dedicated_Sram). "
             "If provided, model must be Arm Ethos-U NPU optimized.",
        default=None
    )

    parser.add_argument(
        "--ethos_u_config",
        type=str,
        help="Arm Ethos-U NPU accelerator config (e.g. ethos-u55-128). "
             "If provided, model must be Arm Ethos-U NPU optimized.",
        default=None
    )

    parser.add_argument(
        "--allow_pte_mac_mismatch",
        action="store_true",
        help="Allow different MAC counts between U85 PTE and firmware configurations.",
    )

    args = parser.parse_args()
    # pylint: enable=duplicate-code

    env = Environment(loader=FileSystemLoader(Path(__file__).parent / 'templates'),
                      trim_blocks=True,
                      lstrip_blocks=True)

    # Resolve and validate input path early to fail fast.
    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise ValueError(f"{model_path} not found")

    suffix = model_path.suffix.lower()

    # Only validate Arm Ethos-U NPU metadata when those args are explicitly provided.
    validation_args = NpuValidationArgs(
        memory_mode=args.ethos_u_memory_mode,
        config=args.ethos_u_config,
    )
    if suffix == ".tflite":
        # TFLite validation runs only when args are set (it will early-return otherwise).
        validate_tflite_model(model_path, validation_args)
    elif suffix == ".pte":
        # PTE validation runs only when args are set (it will early-return otherwise).
        validate_pte_model(
            model_path, validation_args, allow_mac_mismatch=args.allow_pte_mac_mismatch
        )
    else:
        raise ValueError(
            f"Unsupported model type for {model_path}. Expected .tflite or .pte."
        )

    # Output filename mirrors the input model name with a .cc suffix.
    cpp_filename = (Path(args.output_dir) / (model_path.name + ".cc")).resolve()
    print(f"++ Converting {model_path.name} to\
    {cpp_filename.name}")

    cpp_filename.parent.mkdir(exist_ok=True)

    hdr = gen_header(env, args.license_template, model_path.name)

    env \
        .get_template('tflite.cc.template') \
        .stream(common_template_header=hdr,
                model_data=get_model_data(str(model_path)),
                expressions=args.expr,
                additional_headers=args.headers,
                namespaces=args.namespaces).dump(str(cpp_filename))


if __name__ == '__main__':
    main()
