#!/usr/bin/env python3
"""Replace feature_path/coords_path with a bare filename stem in CLIP JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def to_filename(path: str) -> str:
    return path.split("/")[-1].rsplit(".", 1)[0]


def transform(records: list[dict]) -> list[dict]:
    out = []
    for row in records:
        feature_path = row["feature_path"]
        coords_path = row["coords_path"]
        filename = to_filename(feature_path)
        coords_filename = to_filename(coords_path)
        if filename != coords_filename:
            raise ValueError(
                f"feature/coords stem mismatch: {filename!r} vs {coords_filename!r}"
            )
        out.append(
            {
                "caption": row["caption"],
                "project_id": row["project_id"],
                "filename": filename,
            }
        )
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "clip_slidechat_test.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to overwriting --input",
    )
    args = parser.parse_args()
    output = args.output or args.input

    with args.input.open() as f:
        records = json.load(f)

    transformed = transform(records)

    with output.open("w") as f:
        json.dump(transformed, f, indent=4)
        f.write("\n")

    print(f"Wrote {len(transformed)} records to {output}")


if __name__ == "__main__":
    main()
