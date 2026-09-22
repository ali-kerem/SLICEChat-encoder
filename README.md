# SLICEChat Encoder

<p align="center"><a href="https://cyberiada.github.io/SLICEChat/">Project Page</a> | <a href="https://arxiv.org/abs/2609.24894">arXiv</a></p>

Official whole-slide image (WSI) encoder training code for [SLICEChat: Progressive In-Encoder Token Pruning for Whole-Slide Pathology Language Models](https://arxiv.org/abs/2609.24894).

This repository trains the SLICEChat encoder from pre-extracted patch features and their spatial coordinates. For training and inference with the full pathology multimodal LLM, see [SLICEChat](https://github.com/ali-kerem/SLICEChat).

Encoder training has two required stages:

1. **MAE pretraining** trains a vision encoder by reconstructing masked WSI patch features.
2. **CLIP training** initializes the encoder from the MAE checkpoint and aligns slide representations with pathology-report text. The provided run also adds and trains a Cropr token-pruning module and an attention pooler.

First complete an MAE run, then pass its checkpoint to CLIP with `--resume`.

## Requirements

- Linux with an NVIDIA CUDA-capable GPU
- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management

Multi-GPU training uses `torchrun` and NCCL. The supplied launchers use four GPUs; adjust the GPU selection and process count in the scripts for your machine.

## Installation

From the repository root:

```bash
uv sync
```

`uv sync` installs the project and its dependencies from `pyproject.toml`. The provided PyTorch, torchvision, FlashAttention, Mamba, and causal-conv1d versions and wheel sources target the development machine. Adjust them together as needed so they are compatible with your hardware and with each other.

## Dataset preparation

See [data/README.md](data/README.md) for pinned annotation sources, exclusions, manifest generation, TRIDENT feature extraction, and HDF5-to-PyTorch conversion. The guide produces four CLIP training/validation manifests under `data/generated/` and explains the MAE CSVs and paired `features/` and `coords/` directories.

> **Important:** We use earlier, pinned snapshots of these datasets. Both datasets have been updated since then, but we do not consider the later versions because our work began with the earlier snapshots and they more closely reflect the original papers' implementations.

## Training

The scripts in `scripts/` are editable run templates. Edit the arguments inside each file before launching it. See [ARGUMENTS.md](ARGUMENTS.md) for the detailed training options and both base model configurations.

In each launcher, set `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, `run_name`, and `base_log_dir`. Use one process per visible GPU. Rename `run_name` for each experiment so the log directory and its `full.log` correspond to that run.

The paper experiments used seed `0` and `--deterministic`. These settings improve reproducibility but provide limited determinism because Mamba and FlashAttention kernels are intrinsically nondeterministic.

### 1. Run MAE pretraining

Edit `scripts/mae_train.sh`. The main settings are:

- `--data_path`: tensor root containing `features/` and `coords/`.
- `--csv_path`: `data/mae_slideinstruct.csv` or `data/mae_wsillava.csv`.
- `--model_config`: MAE architecture YAML; the provided Mamba `base.yaml` defines the encoder and reconstruction decoder.
- `--batch_size`: slides per GPU per step; the template uses `4`.
- `--epochs` and `--mask_ratio`: training duration and masked-patch fraction; the template uses `30` and `0.4`.

Launch:

```bash
bash scripts/mae_train.sh
```

### 2. Select the MAE checkpoint for CLIP

Keep the checkpoint together with its saved model configuration:

```text
logs/mae/<mae_run>/
├── model_cfg.yaml
└── checkpoints/
    └── checkpoint-29.pth
```

CLIP recovers the vision architecture from this `model_cfg.yaml` and loads the encoder weights from the checkpoint. For the MAE run above, set this argument inside `scripts/clip_train.sh`:

```bash
--resume "logs/mae/slicechat/checkpoints/checkpoint-29.pth"
```

### 3. Run CLIP training

Edit `scripts/clip_train.sh`. The main settings are:

- `--resume`: the MAE checkpoint selected above.
- `--train-data` and `--val-data`: generated caption manifests, such as `data/generated/clip_slideinstruct_train.json` and `data/generated/clip_slideinstruct_val.json`.
- `--train-data-dir` and `--val-data-dir`: tensor roots for the two splits.
- `--batch-size`: slide/text pairs per GPU per step; the template uses `32`.
- `--epochs` and `--lr`: training duration and learning rate; the template uses `20` and `7e-4`.

The remaining template options configure Cropr, attention pooling, and which parameters are trained; see [ARGUMENTS.md](ARGUMENTS.md#clip-training).

Launch:

```bash
bash scripts/clip_train.sh
```

The first CLIP run may download the BiomedBERT text model and tokenizer from Hugging Face. Provide network access or cache those files beforehand.

The current template uses `run_name="slicechat"` and writes to `logs/clip/slicechat/`, including `model_cfg.yaml`, parameters, logs, and `checkpoints/`. CLIP checkpoint numbers count completed epochs, so the final checkpoint of a 20-epoch run is `checkpoints/epoch_20.pt`.

## Logging and batch size

Both launchers capture console output in `full.log` under the run directory. MAE also writes TensorBoard events. CLIP reporting and optional Weights & Biases configuration are documented in [ARGUMENTS.md](ARGUMENTS.md#logging-and-reporting).

Effective batch size is per-GPU batch size × accumulation steps × GPU count. The templates use `4 × 1 × 4 = 16` slides for MAE and `32 × 1 × 4 = 128` slide/text pairs for CLIP. If GPU memory is limited, lower the per-GPU batch size and increase `--accum_iter` (MAE) or `--accum-freq` (CLIP).
