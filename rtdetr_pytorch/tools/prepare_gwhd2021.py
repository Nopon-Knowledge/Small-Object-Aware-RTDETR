#!/usr/bin/env python3
"""
Prepare GWHD 2021 CSV annotations for RT-DETR (COCO format).
"""

import argparse
import csv
import json
import os
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from io import BufferedReader
from typing import Dict, Iterable, List, Optional, Set, Tuple


SPLIT_TO_DIR = {
    "train": "train2017",
    "val": "val2017",
    "test": "test2017",
}


@dataclass
class SplitStats:
    split: str
    csv_rows: int = 0
    no_box_rows: int = 0
    malformed_tokens: int = 0
    duplicate_rows_merged: int = 0
    images_unique: int = 0
    images_written: int = 0
    annotations_written: int = 0
    annotations_deduplicated: int = 0
    boxes_clipped: int = 0
    boxes_dropped_invalid: int = 0
    images_missing_on_disk: int = 0
    images_multi_domain: int = 0


@dataclass
class ImageRecord:
    image_name: str
    domains: Set[str] = field(default_factory=set)
    rows: int = 0
    raw_boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GWHD 2021 CSV annotations to COCO format for RT-DETR.",
    )
    parser.add_argument(
        "--source-root",
        type=str,
        required=True,
        help="Path to gwhd_2021 root (contains competition_*.csv and images/).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output root for COCO-style dataset structure.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to process.",
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        default="symlink",
        choices=["symlink", "hardlink", "copy"],
        help="How to materialize images in output folders.",
    )
    parser.add_argument(
        "--normalize-format",
        action="store_true",
        help="Re-encode all output images as real PNG files (useful for mixed JPEG/PNG content).",
    )
    parser.add_argument(
        "--keep-empty-images",
        action="store_true",
        help="Keep images whose annotation is no_box/empty in COCO images list.",
    )
    parser.add_argument(
        "--category-name",
        type=str,
        default="wheat_head",
        help="Single category name used for generated COCO categories[0].name.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing output split folders/json before writing new results.",
    )
    return parser.parse_args()


def parse_box_token(token: str) -> Optional[Tuple[float, float, float, float]]:
    token = token.strip()
    if not token:
        return None
    parts = token.split()
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(p) for p in parts)
    except ValueError:
        return None
    return (x1, y1, x2, y2)


def parse_boxes(boxes_str: str) -> Tuple[List[Tuple[float, float, float, float]], int, bool]:
    if boxes_str is None:
        return [], 0, True

    text = boxes_str.strip()
    if text == "" or text == "no_box":
        return [], 0, True

    parsed: List[Tuple[float, float, float, float]] = []
    malformed = 0
    for token in text.split(";"):
        box = parse_box_token(token)
        if box is None:
            malformed += 1
            continue
        parsed.append(box)
    return parsed, malformed, False


def load_split_csv(source_root: str, split: str, stats: SplitStats) -> Dict[str, ImageRecord]:
    csv_path = os.path.join(source_root, f"competition_{split}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    records: Dict[str, ImageRecord] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_name", "BoxesString", "domain"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

        for row in reader:
            stats.csv_rows += 1
            image_name = row["image_name"].strip()
            boxes, malformed, is_no_box = parse_boxes(row["BoxesString"])
            if is_no_box:
                stats.no_box_rows += 1
            stats.malformed_tokens += malformed

            domain = row["domain"].strip()
            rec = records.get(image_name)
            if rec is None:
                rec = ImageRecord(image_name=image_name)
                records[image_name] = rec
            else:
                stats.duplicate_rows_merged += 1

            rec.rows += 1
            if domain:
                rec.domains.add(domain)
            rec.raw_boxes.extend(boxes)

    stats.images_unique = len(records)
    stats.images_multi_domain = sum(1 for rec in records.values() if len(rec.domains) > 1)
    return records


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def link_or_copy(src: str, dst: str, mode: str) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def _read_u16be(f: BufferedReader) -> int:
    data = f.read(2)
    if len(data) != 2:
        raise ValueError("Unexpected EOF while reading u16")
    return int.from_bytes(data, "big")


def inspect_image(path: str) -> Tuple[int, int, str]:
    with open(path, "rb") as f:
        sig = f.read(32)
        if len(sig) < 10:
            raise ValueError(f"Invalid image file (too small): {path}")

        # PNG signature + IHDR width/height
        if sig.startswith(b"\x89PNG\r\n\x1a\n"):
            width = int.from_bytes(sig[16:20], "big")
            height = int.from_bytes(sig[20:24], "big")
            return width, height, "PNG"

        # JPEG SOI + scan for SOF marker
        if sig[0:2] == b"\xff\xd8":
            f.seek(0)
            marker_size = 2
            while True:
                f.seek(marker_size, os.SEEK_CUR)
                byte = f.read(1)
                if not byte:
                    break
                while byte == b"\xff":
                    byte = f.read(1)
                    if not byte:
                        break
                if not byte:
                    break

                marker_type = byte[0]
                length_data = f.read(2)
                if len(length_data) != 2:
                    break
                marker_size = struct.unpack(">H", length_data)[0] - 2
                if marker_size < 0:
                    break

                if marker_type in (
                    0xC0,
                    0xC1,
                    0xC2,
                    0xC3,
                    0xC5,
                    0xC6,
                    0xC7,
                    0xC9,
                    0xCA,
                    0xCB,
                    0xCD,
                    0xCE,
                    0xCF,
                ):
                    f.read(1)  # precision
                    height = _read_u16be(f)
                    width = _read_u16be(f)
                    return width, height, "JPEG"

    # Optional fallback if ImageMagick is available.
    try:
        output = subprocess.check_output(
            ["identify", "-format", "%m %w %h", path],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        parts = output.split()
        if len(parts) == 3:
            fmt, width, height = parts
            return int(width), int(height), fmt.upper()
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass

    raise ValueError(f"Unsupported image format for file: {path}")


def write_normalized_png(src: str, dst: str) -> Tuple[int, int, str]:
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "--normalize-format requires Pillow. Install with: pip install pillow"
        ) from e

    with Image.open(src) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        rgb.save(dst, format="PNG")
    return w, h, "PNG"


def clip_box(
    box: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[Optional[Tuple[float, float, float, float]], bool]:
    x1, y1, x2, y2 = box
    nx1 = min(max(x1, 0.0), float(width))
    ny1 = min(max(y1, 0.0), float(height))
    nx2 = min(max(x2, 0.0), float(width))
    ny2 = min(max(y2, 0.0), float(height))
    clipped = (nx1 != x1) or (ny1 != y1) or (nx2 != x2) or (ny2 != y2)
    if nx2 <= nx1 or ny2 <= ny1:
        return None, clipped
    return (nx1, ny1, nx2, ny2), clipped


def to_coco_bbox(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float, float]:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    area = w * h
    return x1, y1, w, h, area


def build_split_coco(
    source_root: str,
    output_root: str,
    split: str,
    records: Dict[str, ImageRecord],
    stats: SplitStats,
    category_name: str,
    link_mode: str,
    normalize_format: bool,
    keep_empty_images: bool,
) -> Dict:
    image_dir_name = SPLIT_TO_DIR[split]
    image_out_dir = os.path.join(output_root, image_dir_name)
    ann_out_dir = os.path.join(output_root, "annotations")
    ensure_dir(image_out_dir)
    ensure_dir(ann_out_dir)

    coco_images: List[Dict] = []
    coco_annotations: List[Dict] = []
    image_id = 1
    ann_id = 1

    for image_name in sorted(records.keys()):
        rec = records[image_name]
        src_image = os.path.join(source_root, "images", image_name)
        if not os.path.isfile(src_image):
            stats.images_missing_on_disk += 1
            continue

        dst_image = os.path.join(image_out_dir, image_name)
        if normalize_format:
            width, height, _ = write_normalized_png(src_image, dst_image)
        else:
            link_or_copy(src_image, dst_image, link_mode)
            width, height, _ = inspect_image(src_image)

        clipped_boxes: List[Tuple[float, float, float, float]] = []
        clipped_set: Set[Tuple[float, float, float, float]] = set()
        for box in rec.raw_boxes:
            clipped, changed = clip_box(box, width, height)
            if changed:
                stats.boxes_clipped += 1
            if clipped is None:
                stats.boxes_dropped_invalid += 1
                continue
            rounded = tuple(round(v, 3) for v in clipped)
            if rounded in clipped_set:
                stats.annotations_deduplicated += 1
                continue
            clipped_set.add(rounded)
            clipped_boxes.append(rounded)

        if not keep_empty_images and len(clipped_boxes) == 0:
            if os.path.lexists(dst_image):
                os.remove(dst_image)
            continue

        domain = ""
        if rec.domains:
            domain = sorted(rec.domains)[0]

        coco_images.append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
                "domain": domain,
            }
        )

        for box in clipped_boxes:
            x, y, w, h, area = to_coco_bbox(box)
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1

        image_id += 1

    stats.images_written = len(coco_images)
    stats.annotations_written = len(coco_annotations)

    coco = {
        "info": {
            "description": f"GWHD 2021 ({split}) converted to COCO",
            "version": "1.0",
            "date_created": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": category_name, "supercategory": "wheat"}],
    }

    ann_path = os.path.join(ann_out_dir, f"instances_{image_dir_name}.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
    return coco


def list_unreferenced_images(source_root: str, all_referenced: Iterable[str]) -> List[str]:
    image_dir = os.path.join(source_root, "images")
    referenced_set = set(all_referenced)
    extra = []
    if not os.path.isdir(image_dir):
        return extra
    for name in os.listdir(image_dir):
        full_path = os.path.join(image_dir, name)
        if os.path.isfile(full_path) and name not in referenced_set:
            extra.append(name)
    return sorted(extra)


def main() -> None:
    args = parse_args()
    source_root = os.path.abspath(args.source_root)
    output_root = os.path.abspath(args.output_root)
    ensure_dir(output_root)

    split_records: Dict[str, Dict[str, ImageRecord]] = {}
    split_stats: Dict[str, SplitStats] = {}

    if args.clean_output:
        for split in args.splits:
            split_dir = os.path.join(output_root, SPLIT_TO_DIR[split])
            if os.path.isdir(split_dir):
                shutil.rmtree(split_dir)
            ann_json = os.path.join(
                output_root, "annotations", f"instances_{SPLIT_TO_DIR[split]}.json"
            )
            if os.path.isfile(ann_json):
                os.remove(ann_json)

    for split in args.splits:
        stats = SplitStats(split=split)
        records = load_split_csv(source_root, split, stats)
        split_records[split] = records
        split_stats[split] = stats

    for split in args.splits:
        build_split_coco(
            source_root=source_root,
            output_root=output_root,
            split=split,
            records=split_records[split],
            stats=split_stats[split],
            category_name=args.category_name,
            link_mode=args.link_mode,
            normalize_format=args.normalize_format,
            keep_empty_images=args.keep_empty_images,
        )

    all_referenced = []
    for split in args.splits:
        all_referenced.extend(split_records[split].keys())
    unreferenced = list_unreferenced_images(source_root, all_referenced)

    report = {
        "source_root": source_root,
        "output_root": output_root,
        "splits": args.splits,
        "link_mode": args.link_mode,
        "normalize_format": args.normalize_format,
        "keep_empty_images": args.keep_empty_images,
        "summary": {split: vars(split_stats[split]) for split in args.splits},
        "source_unreferenced_images_count": len(unreferenced),
        "source_unreferenced_images": unreferenced,
    }

    report_path = os.path.join(output_root, "preprocess_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
