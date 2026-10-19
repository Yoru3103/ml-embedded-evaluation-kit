#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Generate thirteen barcode classes with Zint for grayscale PTQ calibration."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from qrcode_sample_images import render_background, render_symbol

ROOT = Path(__file__).resolve().parents[1]
# These mappings follow the supplied checkpoint's names; IDs 13 and 14 remain unknown.
CLASSES = (
    ("BARCODE", ("CODE128", "CODE39", "EAN13")),
    ("DM", ("DATAMATRIX",)), ("HANXIN", ("HANXIN",)), ("QR", ("QRCODE",)),
    ("PDF", ("PDF417",)), ("AZTEC", ("AZTEC",)), ("CODEONE", ("CODEONE",)),
    ("DOT", ("DOTCODE",)), ("GM", ("GRIDMATRIX",)), ("MAXI", ("MAXICODE",)),
    ("MPDF", ("MICROPDF417",)), ("ULTRA", ("ULTRA",)), ("UPN", ("UPNQR",)),
)


def parse_args() -> argparse.Namespace:
    """Parse dataset size and Zint executable options.

    :returns: Validated generator options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zint", default=shutil.which("zint"), help="Zint executable path")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "resources_downloaded/qrcode/zint_samples_batch1")
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--validation-per-class", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--background-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    if not args.zint or not Path(args.zint).expanduser().is_file():
        parser.error("Install Zint or pass --zint /absolute/path/to/zint")
    if min(args.per_class, args.validation_per_class) <= 0 or args.seed < 0:
        parser.error("Counts must be positive; seed must be nonnegative")
    if args.image_size < 32:
        parser.error("--image-size must be at least 32")
    if not 0 <= args.background_fraction <= 1:
        parser.error("--background-fraction must be in [0,1]")
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new dataset directory")
    args.zint = str(Path(args.zint).expanduser().resolve())
    args.output_dir = args.output_dir.resolve()
    return args


def make_payload(kind: str, rng: np.random.Generator) -> str:
    """Generate bounded ASCII payloads suitable for each barcode family.

    :param kind:   Zint symbology name.
    :param rng:    Per-sample random generator.
    :returns:      Independently generated encoded content.
    """
    if kind == "EAN13":
        return "".join(str(digit) for digit in rng.integers(0, 10, 12))
    length = int(rng.integers(6, 13)) if kind in {"CODE39", "CODE128"} else int(rng.integers(8, 49))
    payload = "".join(rng.choice(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"), length))
    return "UPNQR\n" + payload if kind == "UPNQR" else payload


def encode_symbol(executable: str, kind: str, payload: str, output: Path) -> dict:
    """Invoke Zint without a shell and retain warnings in the sample provenance.

    :param executable:  Zint binary.
    :param kind:        Zint symbology name.
    :param payload:     Encoded text.
    :param output:      Temporary raster output.
    :returns:           Symbology, encoded content, raster checksum and any warning text.
    """
    command = [executable, "-b", kind, "-d", payload, "-o", str(output),
               "--notext", "--quietzones", "--scale=3"]
    if kind == "DATAMATRIX" and len(payload) % 3:
        command.append("--square")
    if kind == "MAXICODE":
        command.append("--mode=4")
    output.unlink(missing_ok=True)
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    # Zint warning statuses 1-4 still produce valid symbols; errors start at 5.
    if result.returncode not in range(5) or not output.is_file():
        raise RuntimeError(f"Zint {kind} failed ({result.returncode}): {result.stderr}")
    return {"symbology": kind, "payload": payload, "warning": result.stderr.strip(),
            "encoder_options": command[7:],
            "symbol_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def save_sample(directory: Path, split: str, stem: str, image: Image.Image,
                 annotation: tuple[int, np.ndarray] | None) -> dict:
    """Save an eight-bit BMP and its matching YOLO quadrilateral annotation.

    :param directory:   Private dataset workspace.
    :param split:       Calibration or validation split.
    :param stem:        Unique sample filename stem.
    :param image:       Final grayscale image.
    :param annotation:  Known class ID and symbol corners, or background.
    :returns:           Relative image path, checksum and annotation metadata.
    """
    path = directory / "images" / split / f"{stem}.bmp"
    label = directory / "labels" / split / f"{stem}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    content = ""
    class_id, polygon = annotation if annotation is not None else (None, None)
    if polygon is not None:
        normalized = polygon / np.array(image.size)
        if not np.isfinite(normalized).all() or (normalized < 0).any() or (normalized > 1).any():
            raise ValueError(f"Symbol annotation lies outside the image: {stem}")
        content = f"{class_id} " + " ".join(f"{value:.8f}" for value in normalized.ravel()) + "\n"
    label.write_text(content, encoding="utf-8")
    return {"image": str(path.relative_to(directory)), "class_id": class_id,
            "polygon_pixels": polygon.tolist() if polygon is not None else None,
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def generate_symbol(directory: Path, split: str, class_id: int,
                     index: int, args: argparse.Namespace) -> dict:
    """Encode and render one independently seeded symbol.

    :param directory:   Private dataset workspace.
    :param split:       Calibration or validation split.
    :param class_id:    Checkpoint class ID.
    :param index:       Sample number within its class.
    :param args:        Generator configuration.
    :returns:           Complete sample provenance and annotation.
    """
    rng = np.random.default_rng(np.random.SeedSequence(
        [args.seed, int(split == "validation"), class_id, index]))
    kind = CLASSES[class_id][1][index % len(CLASSES[class_id][1])]
    payload = make_payload(kind, rng)
    raster = directory / "symbol.bmp"
    record = encode_symbol(args.zint, kind, payload, raster)
    with Image.open(raster) as symbol:
        image, polygon, degradation = render_symbol(
            symbol, rng, args.image_size, CLASSES[class_id][0], scene_profile(index))
    record.update(save_sample(directory, split,
                              f"{class_id:02d}_{CLASSES[class_id][0]}_{index:04d}",
                              image, (class_id, polygon)), degradation=degradation)
    return record


def scene_profile(index: int) -> str:
    """Stratify each class into 70% matched, 20% clear and 10% difficult samples.

    :param index:   Sample number within its class.
    :returns:       Named photometric scene group.
    """
    return "matched" if index % 10 < 7 else "clear" if index % 10 < 9 else "difficult"


def generate_background(directory: Path, split: str, index: int,
                         args: argparse.Namespace) -> dict:
    """Generate an independently seeded negative image with an empty annotation.

    :param directory:   Private dataset workspace.
    :param split:       Calibration or validation split.
    :param index:       Background sample number.
    :param args:        Generator configuration.
    :returns:           Background provenance and checksum.
    """
    rng = np.random.default_rng(np.random.SeedSequence(
        [args.seed, int(split == "validation"), 99, index]))
    image, parameters = render_background(rng, args.image_size, scene_profile(index))
    record = save_sample(directory, split, f"background_{index:04d}", image, None)
    record["degradation"] = parameters
    return record


def generate_split(directory: Path, split: str, count: int,
                    args: argparse.Namespace) -> list[dict]:
    """Generate disjoint symbols and scenes for one dataset split.

    :param directory:   Private dataset workspace.
    :param split:       Calibration or validation.
    :param count:       Samples per known class.
    :param args:        Generator configuration.
    :returns:           Per-image provenance and annotations.
    """
    records = []
    for class_id, (name, _) in enumerate(CLASSES):
        for index in range(count):
            records.append(generate_symbol(directory, split, class_id, index, args))
        print(f"{split}: {class_id:02d} {name}: {count}", flush=True)
    for index in range(round(len(CLASSES) * count * args.background_fraction)):
        records.append(generate_background(directory, split, index, args))
    (directory / "symbol.bmp").unlink()
    return records


def save_preview(directory: Path, records: list[dict], filename: str,
                   profile: str = "matched"):
    """Create a contact sheet with one transformed sample of each known class.

    :param directory:   Dataset workspace.
    :param records:     Calibration image records.
    :param filename:    Preview PNG filename.
    :param profile:     Scene group to show for each class.
    """
    sheet = Image.new("RGB", (256 * 4, 280 * 4), "white")
    drawing = ImageDraw.Draw(sheet)
    for class_id, (name, _) in enumerate(CLASSES):
        record = next(item for item in records if item["class_id"] == class_id
                      and item["degradation"]["profile"] == profile)
        with Image.open(directory / record["image"]) as source:
            tile = ImageOps.contain(source.convert("RGB"), (256, 256))
            polygon = np.array(record["polygon_pixels"]) * np.array(tile.size) / source.size
        ImageDraw.Draw(tile).polygon([tuple(point) for point in polygon],
                                     outline=(0, 180, 0), width=1)
        left, top = class_id % 4 * 256, class_id // 4 * 280
        sheet.paste(tile, (left, top))
        drawing.text((left + 4, top + 258), f"{class_id}: {name}", fill="black")
    sheet.save(directory / filename)


def main():
    """Generate the complete dataset atomically and record reproducible metadata."""
    args = parse_args()
    version = subprocess.run([args.zint, "--version"], check=True,
                             capture_output=True, text=True).stdout.strip()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="zint-samples-", dir=args.output_dir.parent) as work:
        directory = Path(work) / "dataset"
        directory.mkdir()
        records = {
            "calibration": generate_split(directory, "calibration", args.per_class, args),
            "validation": generate_split(directory, "validation",
                                          args.validation_per_class, args)}
        contents = [(item["symbology"], item["payload"])
                    for samples in records.values() for item in samples if "payload" in item]
        if len(contents) != len(set(contents)):
            raise ValueError("Encoded payloads overlap; use a different seed")
        hashes = [{item["image_sha256"] for item in samples} for samples in records.values()]
        if hashes[0].intersection(hashes[1]):
            raise ValueError("Calibration and validation images overlap")
        names = [name for name, _ in CLASSES] + ["class_13", "class_14"]
        report = {"generator": "Zint + grayscale scene rendering", "zint": version,
                  "zint_binary_sha256": hashlib.sha256(Path(args.zint).read_bytes()).hexdigest(),
                  "configuration": {key: str(value) if isinstance(value, Path) else value
                                    for key, value in vars(args).items()},
                  "class_names": names, "generated_class_ids": list(range(13)),
                  "unknown_class_ids": [13, 14], "synthetic_only": True,
                  "scene_mix": {"matched": 0.7, "clear": 0.2, "difficult": 0.1},
                  "background_count_rule": "round(positive_count * background_fraction)",
                  "reference_images": reference_statistics(),
                  "annotation_scope": "generated symbology and projected ink bounds; "
                                      "not confirmation of the checkpoint's training class map",
                  "split_counts": {key: len(value) for key, value in records.items()},
                  "samples": records}
        (directory / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        (directory / "labels.txt").write_text("\n".join(names) + "\n")
        (directory / "dataset.yaml").write_text(
            f"path: {json.dumps(str(args.output_dir))}\ntrain: images/calibration\n"
            f"val: images/validation\nnames: {json.dumps(names)}\n")
        for split, samples in records.items():
            for profile in ("matched", "clear", "difficult"):
                if all(any(item["class_id"] == class_id
                           and item["degradation"]["profile"] == profile for item in samples)
                       for class_id in range(len(CLASSES))):
                    save_preview(directory, samples, f"preview_{split}_{profile}.png", profile)
        directory.rename(args.output_dir)
    print(f"Dataset: {args.output_dir}; {report['split_counts']}", flush=True)


def reference_statistics() -> list[dict]:
    """Record the original BMPs used to select lighting and framing ranges.

    :returns:   Source checksums and grayscale intensity percentiles.
    """
    records = []
    for path in sorted((ROOT / "resources/qrcode/smaples").glob("*.bmp")):
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L"))
        records.append({"path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "intensity_percentiles_5_50_95":
                        np.percentile(pixels, [5, 50, 95]).tolist()})
    return records


if __name__ == "__main__":
    main()
