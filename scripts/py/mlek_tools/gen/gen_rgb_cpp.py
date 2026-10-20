#  SPDX-FileCopyrightText:  Copyright 2021-2024, 2026 Arm Limited and/or
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
Utility script to convert a set of RGB or grayscale images into
corresponding cpp files and a single hpp file referencing the vectors
from the cpp files.
"""
import glob
import math
import os
import typing
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from jinja2 import Environment, FileSystemLoader

from mlek_tools.gen.gen_utils import gen_header


@dataclass
class ImagesParams:
    """
    Template params for Images.hpp and Images.cc
    """
    num_images: int
    image_size: typing.Sequence
    image_array_names: typing.List[str]
    image_filenames: typing.List[str]
    channels: int = 3


def write_hpp_file(
        images_params: ImagesParams,
        header_file_path: Path,
        cc_file_path: Path,
        header_template_file: str,
        env
):
    """
    Write Images.hpp and Images.cc

    @param images_params:           Template params
    @param header_file_path:        Images.hpp path
    @param cc_file_path:            Images.cc path
    @param header_template_file:    Header template file name
    @param env:                     Jinja2 environment
    """
    print(f"++ Generating {header_file_path}")
    hdr = gen_header(env, header_template_file)

    img_w, img_h = images_params.image_size
    image_size = str(img_w * img_h * images_params.channels)

    env \
        .get_template('sample-data/images/images.h.template') \
        .stream(common_template_header=hdr,
                imgs_count=images_params.num_images,
                img_size=image_size,
                var_names=images_params.image_array_names,
                img_width=img_w,
                img_height=img_h) \
        .dump(str(header_file_path))

    env \
        .get_template('sample-data/images/images.c.template') \
        .stream(common_template_header=hdr,
                var_names=images_params.image_array_names,
                img_names=images_params.image_filenames,
                header_filename=os.path.basename(header_file_path)) \
        .dump(str(cc_file_path))

def resize_crop_image(
        original_image: Image.Image,
        image_size: typing.Sequence
) -> np.ndarray:
    """
    Resize and crop input image

    @param original_image:  Image to resize and crop
    @param image_size:      New image size
    @return:                Resized and cropped image
    """
    # IFM size
    ifm_width = image_size[0]
    ifm_height = image_size[1]

    # Aspect ratio resize
    scale_ratio = (float(max(ifm_width, ifm_height))
                   / float(min(original_image.size[0], original_image.size[1])))
    resized_width = int(original_image.size[0] * scale_ratio)
    resized_height = int(original_image.size[1] * scale_ratio)
    resized_image = original_image.resize(
        size=(resized_width, resized_height),
        resample=Image.Resampling.BILINEAR
    )

    # Crop the center of the image
    resized_image = resized_image.crop((
        (resized_width - ifm_width) / 2,  # left
        (resized_height - ifm_height) / 2,  # top
        (resized_width + ifm_width) / 2,  # right
        (resized_height + ifm_height) / 2  # bottom
    ))

    return np.array(resized_image, dtype=np.uint8).flatten()


def letterbox_geometry(
        source_size: typing.Sequence[int],
        image_size: typing.Sequence[int]
) -> tuple[int, int, int, int]:
    """Calculate the resized width, height, and centered padding offsets.

    :param source_size: Source image width and height.
    :param image_size:  Target image width and height.
    :returns:           Resized width, resized height, left offset, top offset.
    """
    source_width, source_height = source_size
    target_width, target_height = image_size
    scale_ratio = min(target_width / source_width, target_height / source_height)
    resized_width = min(target_width, max(1, round(source_width * scale_ratio)))
    resized_height = min(target_height, max(1, round(source_height * scale_ratio)))
    return (resized_width, resized_height, (target_width - resized_width) // 2,
            (target_height - resized_height) // 2)


def resize_letterbox_image(
        original_image: Image.Image,
        image_size: typing.Sequence
) -> np.ndarray:
    """Resize without cropping and pad to the requested width and height.

    :param original_image: Image to resize and pad.
    :param image_size:     New image width and height.
    :returns:             Flattened resized and padded image.
    """
    target_width, target_height = image_size
    resized_width, resized_height, left, top = letterbox_geometry(
        original_image.size, image_size)
    resized_image = original_image.resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    fill = 114 if original_image.mode == "L" else (114, 114, 114)
    output = Image.new(original_image.mode, (target_width, target_height), fill)
    output.paste(resized_image, (left, top))
    return np.array(output, dtype=np.uint8).flatten()


def write_individual_img_cc_file(
        image_data: np.ndarray,
        image_filename: str,
        cc_filename: Path,
        header_template_file: str,
        array_name: str,
        env
):
    """
    Write image.cc

    @param image_data:              Image data
    @param image_filename:          Image file name
    @param cc_filename:             image.cc path
    @param header_template_file:    Header template file name
    @param array_name:              C++ array name
    @param env:                     Jinja2 environment
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    print(f"++ Converting {image_filename} to {cc_filename.name}")

    hdr = gen_header(env, header_template_file, image_filename)

    hex_line_generator = (', '.join(map(hex, sub_arr))
                          for sub_arr in np.array_split(image_data,
                                                        math.ceil(len(image_data) / 20)))
    env \
        .get_template('sample-data/images/image.c.template') \
        .stream(common_template_header=hdr,
                var_name=array_name,
                img_data=hex_line_generator) \
        .dump(str(cc_filename))


def main():
    """
    Convert images
    """
    # pylint: disable=too-many-locals
    # pylint: disable=duplicate-code
    parser = ArgumentParser()

    parser.add_argument(
        "--image_path",
        type=str,
        help="path to images folder or image file to convert."
    )

    parser.add_argument(
        "--package_gen_dir",
        type=str,
        help="path to directory to be generated."
    )

    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        help="Size (width and height) of the converted images."
    )

    parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 3),
        default=3,
        help="Number of sample image channels."
    )

    parser.add_argument(
        "--resize_mode",
        choices=("crop", "letterbox"),
        default="crop",
        help="Crop to fill or preserve the whole image with gray padding."
    )

    parser.add_argument(
        "--license_template",
        type=str,
        help="Header template file",
        default="header_template.txt"
    )

    args = parser.parse_args()
    # pylint: enable=duplicate-code

    env = Environment(loader=FileSystemLoader(Path(__file__).parent / 'templates'),
                      trim_blocks=True,
                      lstrip_blocks=True)

    # Keep the count of the images converted
    image_idx = 0
    image_filenames = []
    image_array_names = []

    if Path(args.image_path).is_dir():
        filepaths = sorted(glob.glob(str(Path(args.image_path) / '**/*.*'), recursive=True))
    elif Path(args.image_path).is_file():
        filepaths = [args.image_path]
    else:
        raise OSError("Directory or file does not exist.")

    for filepath in filepaths:
        filename = Path(filepath).name

        try:
            original_image = Image.open(filepath).convert("L" if args.channels == 1 else "RGB")
        except UnidentifiedImageError:
            print(f"-- Skipping file {filepath} due to unsupported image format.")
            continue

        image_filenames.append(filename)

        # Save the C file
        os.makedirs(args.package_gen_dir, exist_ok=True)

        cc_filename = (Path(args.package_gen_dir) /
                       (Path(filename).stem.replace(" ", "_") + ".c"))
        array_name = "im" + str(image_idx)
        image_array_names.append(array_name)

        if args.resize_mode == "letterbox":
            image_data = resize_letterbox_image(original_image, args.image_size)
        else:
            image_data = resize_crop_image(original_image, args.image_size)
        write_individual_img_cc_file(image_data,
                                     filename,
                                     cc_filename,
                                     args.license_template,
                                     array_name,
                                     env)

        # Increment image index
        image_idx = image_idx + 1

    header_filepath = Path(args.package_gen_dir) / "sample_files.h"
    common_cc_filepath = Path(args.package_gen_dir) / "sample_files.c"

    images_params = ImagesParams(image_idx, args.image_size, image_array_names,
                                image_filenames, args.channels)

    if len(image_filenames) > 0:
        write_hpp_file(
            images_params,
            header_filepath,
            common_cc_filepath,
            args.license_template,
            env,
        )
    else:
        raise FileNotFoundError("No valid images found.")


if __name__ == '__main__':
    main()
