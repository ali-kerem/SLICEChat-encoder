# SLICEChat Encoder dataset preparation

This guide covers the pinned annotation sources, curated exclusions, caption manifest generation, and patch features used by both encoder training stages.

## Frozen upstream sources

| Dataset | Revision | Files used by the encoder |
| --- | --- | --- |
| [SlideChat](https://huggingface.co/datasets/General-Medical-AI/SlideChat/tree/975a73561ab8ff93455f9d6f2e5a571f940dc9c7) | `975a73561ab8ff93455f9d6f2e5a571f940dc9c7` | `SlideInstruct_train_stage1_caption.json`, `SlideBench-Caption-TCGA.csv` |
| [WSI-LLaVA](https://github.com/XinhengLyu/WSI-LLaVA/tree/7a98bbcb8807a4396a4b971fc7ffaf05cb8f7f90) | `7a98bbcb8807a4396a4b971fc7ffaf05cb8f7f90` | `dataset/1_Report_train.json`, `dataset/1_Report_test.json` |

`SlideInstruct_train_stage1_caption.json` supplies the SlideInstruct training captions. `SlideBench-Caption-TCGA.csv` is used as its held-out validation split. The WSI-LLaVA report train and test source files supply the training and validation splits, respectively.

## Filename mapping and exclusions

SlideChat's source annotations use short slide identifiers, so `data/slidechat_filename_map.csv` records the corresponding filenames used by the encoder manifests.

The curated exclusion lists are:

- `data/dataset_exclusions/missing_mpp_slide_ids.txt`
- `data/dataset_exclusions/invalid_slide_ids.txt`

Together they contain the union required to reproduce all four caption manifests: 69 slides excluded because usable MPP metadata was unavailable and 205 slides excluded because their identifiers were invalid. This includes validation-only exclusions.

## Rebuilding the caption manifests

By default, `data/create_clip_manifests.py`:

1. downloads missing annotation files from the frozen revisions above into `data/source_annotations/<source>/<revision>/<upstream-path>`;
2. loads both curated files under `data/dataset_exclusions/`;
3. resolves SlideChat's short barcodes with `data/slidechat_filename_map.csv`;
4. retains the first `gpt` response, and removes `Microscopic observation of the pathology slide reveals` from the start of WSI-LLaVA captions;
5. filters the curated exclusions and writes four JSON manifests under `data/generated/`.

From the repository root, activate the environment and run the builder:

```bash
source .venv/bin/activate
python data/create_clip_manifests.py
```

To use local annotations and a different output directory:

```bash
python data/create_clip_manifests.py \
    --slidechat-source-dir /path/to/slidechat-annotations \
    --wsillava-source-dir /path/to/wsillava-annotations \
    --no-download-source-files \
    --output-dir /path/to/manifests
```

Both source overrides default to `None`, using the revision-based cache above. Custom roots contain the upstream paths: SlideChat files directly under its root, and WSI-LLaVA files under `dataset/`. Use `--source-dir` to relocate the shared cache instead. Default paths are relative to the script's location; relative CLI paths are resolved from the current working directory.

Existing source files are reused without being overwritten. Local annotations must come from the pinned revisions above. The filename map and exclusion lists still come from this repository's `data/` directory.

To explicitly replace generated manifests on a subsequent run:

```bash
python data/create_clip_manifests.py --overwrite-outputs
```

Use `python data/create_clip_manifests.py --help` to see all options.

The generated files are:

| Manifest | Records |
| --- | ---: |
| `generated/clip_slideinstruct_train.json` | 4,152 |
| `generated/clip_slideinstruct_val.json` | 725 |
| `generated/clip_wsillava_train.json` | 9,480 |
| `generated/clip_wsillava_val.json` | 204 |

The upstream `1_Report_test.json` supplies WSI-LLaVA validation; generated validation manifests use the `_val` suffix. The builder writes only the four JSON manifests and checks all expected record counts before writing them. If any output already exists, it stops unless `--overwrite-outputs` is passed.

Each CLIP record contains a caption and a filename stem without a directory or `.pt` suffix:

```json
[
  {
    "caption": "Pathology report text.",
    "filename": "slide-filename"
  }
]
```

## Extracting patch embeddings

Use [TRIDENT](https://github.com/mahmoodlab/Trident) to segment the WSIs, extract non-overlapping 512-pixel patches at 20x, and encode them with CONCH v1.5. Install TRIDENT by following its repository instructions, then run this command from the TRIDENT repository:

```bash
python run_batch_of_slides.py \
    --task all \
    --wsi_dir /path/to/wsis \
    --job_dir /path/to/trident_output \
    --overlap 0 \
    --patch_size 512 \
    --mag 20 \
    --patch_encoder conch_v15
```

The relevant HDF5 files are written to:

```text
/path/to/trident_output/20x_512px_0px_overlap/features_conch_v15/
```

### Convert HDF5 outputs to PyTorch tensors

Run from this repository's root after installation. `h5py` is included in the project dependencies installed by `uv sync`:

```bash
source .venv/bin/activate
python data/h5_to_pt.py \
    --h5-files-dir /path/to/trident_output/20x_512px_0px_overlap/features_conch_v15 \
    --save-dir /path/to/wsi-tensors \
    --patch-size 512 \
    --num-workers 4
```

| Argument | Description | Default |
| --- | --- | --- |
| `--h5-files-dir` | Directory containing TRIDENT's `.h5` files. | Required |
| `--save-dir` | Output root for `features/` and `coords/`. | Required |
| `--patch-size` | Patch size in pixels used during extraction. | `512` |
| `--num-workers` | Number of parallel conversion processes. | `4` |

The converter preserves patch features, converts pixel coordinates to patch-grid indices, and normalizes their common grid spacing, following the encoder's coordinate convention. `--patch-size` must match the value passed to TRIDENT. Slides with both output tensors already present are skipped.

## Tensor format

Both stages consume pre-extracted patch tensors. The converter creates one feature file and one coordinate file per slide:

```text
/path/to/wsi-tensors/
├── features/
│   ├── slide-a.pt
│   └── slide-b.pt
└── coords/
    ├── slide-a.pt
    └── slide-b.pt
```

Feature tensors have shape `[num_patches, feature_dim]`; coordinate tensors have shape `[num_patches, 2]`, with rows in the same patch order. The default CONCH v1.5 features have dimension `768`.

Set `--data_path` in `scripts/mae_train.sh` or `--train-data-dir` and `--val-data-dir` in `scripts/clip_train.sh` to this tensor root.

## MAE manifests

MAE expects a CSV with a `filename` column containing stems without the `.pt` suffix. The loader adds the suffix when locating both tensors:

```csv
filename
slide-a
slide-b
```
