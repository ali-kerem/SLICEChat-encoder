#!/usr/bin/env python3
"""Create the four CLIP manifests from frozen upstream annotations.

Run with --help for annotation cache, source override, and output options.
This builder does not inspect SVS files, read MPP metadata, query GDC, or create
audit tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPOSITORY_DIR / "data" / "source_annotations"
OUTPUT_DIR = REPOSITORY_DIR / "data" / "generated"
EXCLUSION_DIR = REPOSITORY_DIR / "data" / "dataset_exclusions"
SLIDECHAT_FILENAME_MAP = REPOSITORY_DIR / "data" / "slidechat_filename_map.csv"

REQUEST_TIMEOUT_SECONDS = 60

SLIDECHAT_REVISION = "975a73561ab8ff93455f9d6f2e5a571f940dc9c7"
WSILLAVA_REVISION = "7a98bbcb8807a4396a4b971fc7ffaf05cb8f7f90"

SOURCE_URLS = {
    "SlideInstruct_train_stage1_caption.json": (
        "https://huggingface.co/datasets/General-Medical-AI/SlideChat/resolve/"
        f"{SLIDECHAT_REVISION}/SlideInstruct_train_stage1_caption.json?download=true"
    ),
    "SlideBench-Caption-TCGA.csv": (
        "https://huggingface.co/datasets/General-Medical-AI/SlideChat/resolve/"
        f"{SLIDECHAT_REVISION}/SlideBench-Caption-TCGA.csv?download=true"
    ),
    "1_Report_train.json": (
        "https://raw.githubusercontent.com/XinhengLyu/WSI-LLaVA/"
        f"{WSILLAVA_REVISION}/dataset/1_Report_train.json"
    ),
    "1_Report_test.json": (
        "https://raw.githubusercontent.com/XinhengLyu/WSI-LLaVA/"
        f"{WSILLAVA_REVISION}/dataset/1_Report_test.json"
    ),
}

OUTPUT_NAMES = {
    ("slideinstruct", "train"): "clip_slideinstruct_train.json",
    ("slideinstruct", "val"): "clip_slideinstruct_val.json",
    ("wsillava", "train"): "clip_wsillava_train.json",
    ("wsillava", "val"): "clip_wsillava_val.json",
}

EXPECTED_COUNTS = {
    ("slideinstruct", "train"): 4_152,
    ("slideinstruct", "val"): 725,
    ("wsillava", "train"): 9_480,
    ("wsillava", "val"): 204,
}

MISSING_MPP_FILE = EXCLUSION_DIR / "missing_mpp_slide_ids.txt"
INVALID_SLIDES_FILE = EXCLUSION_DIR / "invalid_slide_ids.txt"
WSILLAVA_CAPTION_PREFIX = re.compile(
    r"^Microscopic observation of the pathology slide reveals(?:\.\.\.|[:,])?\s*"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=SOURCE_DIR,
        help="Base annotation cache (default: data/source_annotations).",
    )
    parser.add_argument(
        "--slidechat-source-dir", type=Path, default=None,
        help="SlideChat annotation root; defaults to <source-dir>/slidechat/<revision>.",
    )
    parser.add_argument(
        "--wsillava-source-dir", type=Path, default=None,
        help="WSI-LLaVA root containing dataset/; defaults to <source-dir>/wsillava/<revision>.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Manifest output directory (default: data/generated).",
    )
    parser.add_argument(
        "--download-source-files", action=argparse.BooleanOptionalAction,
        default=True, help="Download missing annotations (default: enabled).",
    )
    parser.add_argument(
        "--overwrite-outputs", action=argparse.BooleanOptionalAction,
        default=False, help="Replace existing generated manifests (default: disabled).",
    )
    return parser.parse_args(argv)


def source_paths(args: argparse.Namespace) -> dict[str, Path]:
    slidechat_root = args.slidechat_source_dir
    if slidechat_root is None:
        slidechat_root = args.source_dir / "slidechat" / SLIDECHAT_REVISION
    wsillava_root = args.wsillava_source_dir
    if wsillava_root is None:
        wsillava_root = args.source_dir / "wsillava" / WSILLAVA_REVISION
    return {
        "SlideInstruct_train_stage1_caption.json": (
            slidechat_root / "SlideInstruct_train_stage1_caption.json"
        ),
        "SlideBench-Caption-TCGA.csv": slidechat_root / "SlideBench-Caption-TCGA.csv",
        "1_Report_train.json": wsillava_root / "dataset" / "1_Report_train.json",
        "1_Report_test.json": wsillava_root / "dataset" / "1_Report_test.json",
    }


def download_file(url: str, destination: Path) -> None:
    if destination.is_file():
        print(f"Using existing {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "SLICEChat-encoder-dataset-prep"})
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Downloaded {destination}")


def download_sources(paths: dict[str, Path], enabled: bool) -> None:
    if not enabled:
        return
    for filename, url in SOURCE_URLS.items():
        download_file(url, paths[filename])


def normalized_slide_id(value: str) -> str:
    name = Path(value.strip()).name
    lower_name = name.lower()
    for suffix in (".svs", ".pt", ".csv", ".h5"):
        if lower_name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def slide_barcode(value: str) -> str:
    return normalized_slide_id(value).split(".", maxsplit=1)[0]


def has_exact_slide_filename(value: str) -> bool:
    return "." in normalized_slide_id(value)


def load_exclusion_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing exclusion file: {path}")
    with path.open(encoding="utf-8") as file:
        values = set()
        for line in file:
            value = line.partition("#")[0].strip()
            if value:
                values.add(normalized_slide_id(value).lower())
        return values


def is_excluded(slide_id: str, exclusion_ids: set[str]) -> bool:
    normalized = normalized_slide_id(slide_id).lower()
    if normalized in exclusion_ids:
        return True
    if has_exact_slide_filename(slide_id):
        return False
    barcode = slide_barcode(slide_id).lower()
    return any(slide_barcode(value).lower() == barcode for value in exclusion_ids)


def load_slidechat_filename_map(path: Path) -> dict[str, str]:
    mapping = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            barcode = slide_barcode(row["slide_id"]).lower()
            filename = normalized_slide_id(row["filename"])
            if barcode in mapping and mapping[barcode] != filename:
                raise ValueError(f"Ambiguous SlideChat filename mapping for {barcode}")
            mapping[barcode] = filename
    return mapping


def first_gpt_response(conversations: Iterable[dict[str, Any]]) -> str:
    for message in conversations:
        if str(message.get("from", "")).lower() == "gpt":
            return str(message.get("value", "")).strip()
    return ""


def conversation_records(path: Path, strip_wsillava_prefix: bool) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        source_rows = json.load(file)

    records = []
    for index, row in enumerate(source_rows):
        image = row.get("image", "")
        if isinstance(image, list):
            image = image[0] if image else ""
        filename = normalized_slide_id(str(image))
        caption = first_gpt_response(row.get("conversations", []))
        if strip_wsillava_prefix:
            caption = WSILLAVA_CAPTION_PREFIX.sub("", caption)
        if not filename or not caption:
            raise ValueError(f"Incomplete record in {path.name} at row {index}")
        records.append({"caption": caption, "filename": filename})
    return records


def slidebench_records(path: Path) -> list[dict[str, str]]:
    records = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row_number, row in enumerate(reader, start=2):
            if len(row) < 5:
                raise ValueError(f"Expected at least five columns in {path}:{row_number}")
            filename = normalized_slide_id(row[1])
            caption = re.sub(r"\s*\n+\s*", " ", row[4]).strip()
            if not filename or not caption:
                raise ValueError(f"Incomplete record in {path}:{row_number}")
            records.append({"caption": caption, "filename": filename})
    return records


def load_source_records(
    paths: dict[str, Path],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source annotation files:\n" + "\n".join(missing))

    return {
        ("slideinstruct", "train"): conversation_records(
            paths["SlideInstruct_train_stage1_caption.json"], False
        ),
        ("slideinstruct", "val"): slidebench_records(
            paths["SlideBench-Caption-TCGA.csv"]
        ),
        ("wsillava", "train"): conversation_records(
            paths["1_Report_train.json"], True
        ),
        ("wsillava", "val"): conversation_records(paths["1_Report_test.json"], True),
    }


def convert_records(
    dataset: str,
    records: list[dict[str, str]],
    exclusions: set[str],
    slidechat_filenames: dict[str, str],
) -> tuple[list[dict[str, str]], int]:
    converted = []
    excluded_count = 0
    for record in records:
        source_id = record["filename"]
        if is_excluded(source_id, exclusions):
            excluded_count += 1
            continue

        filename = source_id
        if dataset == "slideinstruct":
            barcode = slide_barcode(source_id).lower()
            try:
                filename = slidechat_filenames[barcode]
            except KeyError as error:
                raise KeyError(
                    f"No full filename mapping for unexcluded SlideChat ID {source_id}"
                ) from error
        converted.append({"caption": record["caption"], "filename": filename})
    return converted, excluded_count


def write_json(
    path: Path, records: list[dict[str, str]], *, overwrite: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)
        file.write("\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.overwrite_outputs:
        existing = [
            str(args.output_dir / name)
            for name in OUTPUT_NAMES.values()
            if (args.output_dir / name).exists()
        ]
        if existing:
            raise FileExistsError(
                f"Outputs already exist: {existing}; choose another --output-dir "
                "or use --overwrite-outputs"
            )

    paths = source_paths(args)
    download_sources(paths, args.download_source_files)
    sources = load_source_records(paths)
    exclusions = load_exclusion_ids(MISSING_MPP_FILE)
    exclusions.update(load_exclusion_ids(INVALID_SLIDES_FILE))
    slidechat_filenames = load_slidechat_filename_map(SLIDECHAT_FILENAME_MAP)

    outputs = {}
    for key, records in sources.items():
        converted, excluded_count = convert_records(
            key[0], records, exclusions, slidechat_filenames
        )
        expected_count = EXPECTED_COUNTS[key]
        if len(converted) != expected_count:
            raise ValueError(
                f"{key[0]} {key[1]} produced {len(converted)} records; "
                f"expected {expected_count}"
            )
        outputs[key] = (converted, excluded_count)

    for key, (converted, excluded_count) in outputs.items():
        destination = args.output_dir / OUTPUT_NAMES[key]
        write_json(destination, converted, overwrite=args.overwrite_outputs)
        print(
            f"{key[0]} {key[1]}: wrote {len(converted)} records to {destination} "
            f"({excluded_count} excluded)"
        )


if __name__ == "__main__":
    main()
