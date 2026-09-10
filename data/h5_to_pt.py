import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce
from pathlib import Path

import h5py
import torch
from tqdm import tqdm


def positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TRIDENT HDF5 features and coordinates to PyTorch files."
    )
    parser.add_argument(
        "--h5-files-dir",
        type=Path,
        required=True,
        help="Directory containing the TRIDENT .h5 files.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="Output directory in which features/ and coords/ will be created.",
    )
    parser.add_argument(
        "--patch-size",
        type=positive_int,
        default=512,
        help="Patch size in pixels used during extraction (default: 512).",
    )
    parser.add_argument(
        "--num-workers",
        type=positive_int,
        default=4,
        help="Number of parallel conversion processes (default: 4).",
    )
    return parser.parse_args()


def convert_file(
    file: str,
    coords_save_dir: str,
    features_save_dir: str,
    patch_size: int,
) -> str:
    slide_name = os.path.splitext(os.path.basename(file))[0] + ".pt"
    coords_save_path = os.path.join(coords_save_dir, slide_name)
    features_save_path = os.path.join(features_save_dir, slide_name)

    if os.path.exists(coords_save_path) and os.path.exists(features_save_path):
        return f"skipped {slide_name}"

    with h5py.File(file, "r") as h5_file:
        coords = torch.from_numpy(h5_file["coords"][:])
        features = torch.from_numpy(h5_file["features"][:])

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"Expected coords with shape [num_patches, 2], got {tuple(coords.shape)}"
        )
    if coords.shape[0] == 0:
        raise ValueError("Cannot convert an HDF5 file with no coordinates")
    if features.shape[0] != coords.shape[0]:
        raise ValueError(
            "Features and coordinates must contain the same number of patches: "
            f"got {features.shape[0]} features and {coords.shape[0]} coordinates"
        )

    min_coords = torch.min(coords, dim=0).values
    indexes = (coords - min_coords) / patch_size

    gcds = reduce(torch.gcd, indexes.long())
    if gcds[0] == gcds[1] and gcds[0] > 1:
        indexes = indexes / gcds[0]

    torch.save(indexes, coords_save_path)
    torch.save(features, features_save_path)
    return f"converted {slide_name}"


def main() -> None:
    args = parse_args()
    coords_save_dir = args.save_dir / "coords"
    features_save_dir = args.save_dir / "features"
    coords_save_dir.mkdir(parents=True, exist_ok=True)
    features_save_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(args.h5_files_dir.glob("*.h5"))

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                convert_file,
                str(file),
                str(coords_save_dir),
                str(features_save_dir),
                args.patch_size,
            ): file
            for file in h5_files
        }
        with tqdm(total=len(futures), unit="file") as progress:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    source_file = os.path.basename(futures[future])
                    tqdm.write(f"ERROR on {source_file}: {error}")
                finally:
                    progress.update(1)


if __name__ == "__main__":
    main()
