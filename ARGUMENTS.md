# Argument reference

Values labeled **Template** come from the supplied shell scripts or YAML files. Values marked **default** come from the Python parser when the launcher omits that option. Edit arguments inside the shell scripts; the launchers do not forward extra command-line arguments.

## Launchers

| Setting | Meaning | MAE / CLIP template |
| --- | --- | --- |
| `run_name` | Experiment name and final component of the log directory. Rename for each run. | `slicechat` / `slicechat` |
| `base_log_dir` | Base directory containing experiment folders. | `logs/mae` / `logs/clip` |
| `CUDA_VISIBLE_DEVICES` | Physical GPUs made visible to the run. | `0,1,2,3` |
| `NPROC_PER_NODE` | Training processes on this machine; use one per visible GPU. | `4` / `4` |

The provided launchers configure a single node and select a local rendezvous port automatically.

## MAE training

Edit [scripts/mae_train.sh](scripts/mae_train.sh).

### Data and optimization

| Argument | Meaning | Template or default |
| --- | --- | --- |
| `--model_config` | Architecture YAML for the encoder, decoder, and reconstruction loss. | [Mamba base.yaml](src/slicechat_encoder/models/mamba/model_configs/base.yaml) |
| `--data_path` | Tensor root containing `features/` and `coords/`. | Set your tensor root |
| `--csv_path` | CSV with a `filename` column and no `.pt` suffix. | Set `data/mae_slideinstruct.csv` or `data/mae_wsillava.csv` |
| `--batch_size` | Slides per GPU per forward pass. | `4` |
| `--accum_iter` | Micro-batches accumulated per optimizer update. | `1` (default) |
| `--epochs` | Total number of training epochs. | `30` |
| `--mask_ratio` | Fraction of valid patch tokens masked for reconstruction. | `0.4` |
| `--lr` | Absolute learning rate; overrides batch-scaled `--blr` when supplied. | Unset |
| `--blr` | Base learning rate, scaled by effective batch size divided by 256. | `1e-3` (default) |
| `--min_lr` | Final learning rate of cosine decay. | `0.0` (default) |
| `--weight_decay` | AdamW weight decay. | `0.05` (default) |
| `--warmup_ratio` | Fraction of epochs used for learning-rate warmup, rounded down to whole epochs. | `0.1` (default) |
| `--num_workers` | Data-loader workers per training process. | `4` |
| `--pin_mem` / `--no_pin_mem` | Enable or disable pinned data-loader memory. | Enabled (default) |
| `--device` | Device used for training. | `cuda` (default) |
| `--seed` | Base random seed; each distributed rank adds its rank. | `0` |
| `--deterministic` | Disables cuDNN benchmarking and requests deterministic algorithms with warnings. | Enabled |
| `--distributed` | Initializes distributed training from the launch environment. | Enabled |

Effective batch size is `batch_size × accum_iter × world_size`. With the template's four GPUs this is `4 × 1 × 4 = 16`, so the derived learning rate is `1e-3 × 16 / 256 = 6.25e-5`. AdamW uses betas `(0.9, 0.95)`; warmup lasts `int(warmup_ratio × epochs)` epochs, followed by cosine decay.

### Checkpoints

| Argument | Meaning | Template or default |
| --- | --- | --- |
| `--output_dir` | Run directory for `model_cfg.yaml`, `log.txt`, and `checkpoints/`. | `logs/mae/<run_name>` |
| `--log_dir` | TensorBoard event directory. | `logs/mae/<run_name>` |
| `--save_interval` | Checkpoint interval in zero-based epochs; epoch zero and the final epoch are saved. | `10` (default) |
| `--resume` | MAE checkpoint for restoring model, optimizer, scaler, and epoch state. | Empty (default) |
| `--start_epoch` | Starting epoch when not restored from a checkpoint. | `0` (default) |

MAE checkpoints use `checkpoint-<epoch>.pth`; a 30-epoch run ends with `checkpoint-29.pth`. Keep the saved `model_cfg.yaml` one directory above `checkpoints/` when transferring the encoder to CLIP.

## CLIP training

Edit [scripts/clip_train.sh](scripts/clip_train.sh).

### Model initialization and trainable components

| Argument | Meaning | Template |
| --- | --- | --- |
| `--resume` | MAE checkpoint used to initialize the provided pruner-training run. | `logs/mae/slicechat/checkpoints/checkpoint-29.pth` |
| `--model` | Base CLIP configuration for the text tower and shared embedding dimension. | `base` |
| `--pruner-train` | Parameter-name substrings allowed to train; all other parameters are frozen. | `cropr visual.pooler text.proj` |
| `--pruner-train-cropr-cfg` | YAML configuring the Cropr modules added to the vision tower. | [Cropr base.yaml](src/slicechat_encoder/models/token_compressors/cropr_configs/base.yaml) |
| `--pruner-train-pooler-type` | Visual pooling method: `cls`, `average`, `attention`, or `unchanged`. | `attention` |
| `--pruner-train-pruning-rate` | Overall fraction of visual tokens to remove, in `[0, 1)`. | `0.8` |
| `--lock-text` | Freezes the Hugging Face Transformer backbone; the separate text projection can still train. | Enabled |
| `--lock-text-freeze-layer-norm` | Also freezes LayerNorm parameters in the locked text backbone. | Enabled |

With an MAE checkpoint, CLIP loads its saved `encoder_config` as `vision_cfg`, combines it with the selected CLIP base configuration, and applies the pruner-training overrides. The resulting configuration is saved to the new run's `model_cfg.yaml`.

Resuming an interrupted CLIP training run is currently unsupported.

`--grad-checkpointing` is disabled by the training entry point during pruner training because that combination causes gradient problems.

### Data and optimization

| Argument | Meaning | Template or default |
| --- | --- | --- |
| `--train-data` / `--val-data` | Training and validation caption JSONs. | Select the generated `clip_slideinstruct_*` or `clip_wsillava_*` manifests |
| `--train-data-dir` / `--val-data-dir` | Tensor roots for the corresponding splits; may be the same directory. | Set your tensor roots |
| `--batch-size` | Slide/text pairs per GPU per forward pass. | `32` |
| `--accum-freq` | Micro-batches accumulated per optimizer update. | `1` |
| `--epochs` | Total number of training epochs. | `20` |
| `--lr` | Optimizer learning rate. | `7e-4` |
| `--wd` | Weight decay. | `0.1` |
| `--opt` | Optimizer implementation. | `adamw` (default) |
| `--warmup` | Fraction of total optimizer steps used for warmup. | `0.1` |
| `--lr-scheduler` | Learning-rate schedule: `cosine`, `const`, or `const-cooldown`. | `cosine` (default) |
| `--precision` | Numerical precision mode. | `bf16` |
| `--workers` | Data-loader workers per GPU/process. | `4` |
| `--seed` | Base random seed. | `0` |
| `--deterministic` | Requests reproducible algorithm settings. | Enabled |

Effective batch size is `batch_size × accum_freq × world_size`: `32 × 1 × 4 = 128` slide/text pairs with the template. Unlike MAE's `--blr`, the provided CLIP `--lr` is an absolute learning rate.

The paper used seed `0`. `--deterministic` provides limited determinism: Mamba and FlashAttention kernels can still produce nondeterministic results.

### Validation and checkpoints

| Argument | Meaning | Template or default |
| --- | --- | --- |
| `--val-frequency` | Evaluate `--val-data` every N epochs and at the final epoch; `0` disables it. | `1` (default) |
| `--save-frequency` | Save numbered checkpoints every N completed epochs; the final epoch is also saved. | `1` (default) |
| `--delete-previous-checkpoint` | Removes the preceding epoch's numbered checkpoint. | Enabled |
| `--save-best-only` | Save only improvements in validation loss as `epoch_best.pt`. | Disabled (default) |
| `--save-most-recent` | Also maintain `epoch_latest.pt` when best-only saving is disabled. | Disabled (default) |
| `--name` | Experiment folder below `--log-dir`. | `<run_name>` |
| `--log-dir` | Base output directory. | `logs/clip` |

CLIP checkpoints use completed-epoch numbers: a 20-epoch run ends with `epoch_20.pt`. Preserve the run's `model_cfg.yaml` above `checkpoints/`.

## MAE model configuration

File: [src/slicechat_encoder/models/mamba/model_configs/base.yaml](src/slicechat_encoder/models/mamba/model_configs/base.yaml).

| Top-level key | Meaning | Template |
| --- | --- | --- |
| `feature_dim` | Input patch-feature dimension and reconstruction output dimension. | `768` |
| `norm_feature_loss` | Standardizes each target patch vector before masked-feature MSE. | `false` |
| `encoder_config` | Vision backbone used by MAE and later transferred to CLIP. | 768-dimensional hybrid Mamba |
| `decoder_config` | Reconstruction network, used only during MAE training. | 512-dimensional hybrid Mamba |

Input features go directly into the encoder, so `feature_dim` must match `encoder_config.encoder_embed_dim`. A learned projection connects the encoder to the decoder, allowing a different decoder width. The loss averages MSE over masked, valid patches.

### Encoder and decoder settings

The following keys occur under both `encoder_config` and `decoder_config`.

| Key | Meaning | Encoder / decoder template |
| --- | --- | --- |
| `encoder_embed_dim` | Hidden width of this network. | `768` / `512` |
| `block_structure` | Repeated layer pattern: `M` for Mamba, `T` for Transformer. | `MMMT` / `MMMT` |
| `num_blocks` | Number of repetitions of the layer pattern. | `6` / `2` |
| `d_state` | Mamba state dimension. | `16` / `16` |
| `d_conv` | Mamba local convolution width. | `4` / `4` |
| `expand` | Mamba inner-width expansion factor. | `2` / `2` |
| `dropout` | Dropout applied to input token embeddings. | `0.0` / `0.0` |
| `drop_path_rate` | Stochastic-depth rate used by the network. | `0.1` / `0.05` |
| `rms_norm` | Select RMSNorm for Mamba/final normalization when available. | `true` / `true` |
| `layernorm_eps` | Epsilon for Mamba/final normalization. | `1e-5` / `1e-5` |
| `normalize_output` | Apply the final normalization layer; this is not L2 normalization. | `true` / `true` |
| `residual_in_fp32` | Accumulate residuals in float32. | `true` / `true` |
| `fused_add_norm` | Use fused residual-add/normalization kernels when available. | `false` / `false` |
| `biscan` | Enable bidirectional Mamba scanning. | `true` / `true` |
| `pos_embed` | Spatial embedding: `rope_2d` for attention or `sincos_2d` added to tokens. | `rope_2d` / `rope_2d` |
| `rope_2d_theta` | Frequency-base parameter for 2D rotary embeddings. | `150.0` / `150.0` |
| `rope_2d_mixed` | Use mixed, learnable 2D rotary frequencies instead of fixed axial frequencies. | `true` / `true` |
| `pooler` | Slide pooling method: `average`, `attention`, or `cls`. MAE reconstruction uses token outputs. | `average` / `average` |
| `normalize_image_patches` | L2-normalize input token vectors before the network. | `false` / `false` |

### Hybrid Transformer settings

Keys below are relative to `encoder_config.hybrid_vit_config` or `decoder_config.hybrid_vit_config`.

| Key | Meaning | Encoder / decoder template |
| --- | --- | --- |
| `num_attention_heads` | Attention heads; must divide the network's hidden width. | `12` / `8` |
| `qkv_bias` | Add bias to query, key, and value projections. | `true` / `true` |
| `mlp_ratio` | Transformer feed-forward width divided by hidden width. | `4.0` / `4.0` |
| `hidden_act` | Feed-forward activation. | `gelu_pytorch_tanh` / `gelu_pytorch_tanh` |
| `layer_norm_eps` | Transformer LayerNorm epsilon. | `1e-6` / `1e-6` |
| `attention_dropout` | Attention-probability dropout during training. | `0.0` / `0.0` |
| `hidden_dropout` | Transformer hidden-state dropout. | `0.0` / `0.0` |

The hybrid hidden size is derived from the containing network's `encoder_embed_dim`. Set attention heads and the feed-forward expansion under `hybrid_vit_config` for each network.

## CLIP model configuration

File: [src/slicechat_encoder/train/open_clip/model_configs/base.yaml](src/slicechat_encoder/train/open_clip/model_configs/base.yaml), selected with `--model base`.

| Key | Meaning | Template |
| --- | --- | --- |
| `embed_dim` | Shared slide/text embedding dimension. Must match the loaded vision encoder's output width. | `768` |
| `custom_text` | Uses the separate Hugging Face text-tower implementation. | `true` |
| `text_cfg.hf_model_name` | Hugging Face model ID or local model directory for the text backbone. | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` |
| `text_cfg.hf_tokenizer_name` | Matching tokenizer model ID or local directory. | Same BiomedBERT model |
| `text_cfg.hf_proj_type` | Projection from text hidden width to shared embedding width. | `linear` |
| `text_cfg.hf_pooler_type` | Text pooling strategy; `cls_pooler` uses the model's pooled CLS output when available. | `cls_pooler` |

The text model is pretrained by default (`hf_model_pretrained: true`, omitted from the YAML). The base file intentionally has no `vision_cfg`: the MAE checkpoint supplies it. When loading a saved CLIP run, that run's configuration supplies both towers instead.

### Cropr configuration

File: [src/slicechat_encoder/models/token_compressors/cropr_configs/base.yaml](src/slicechat_encoder/models/token_compressors/cropr_configs/base.yaml), selected with `--pruner-train-cropr-cfg`.

| Key | Meaning | Template |
| --- | --- | --- |
| `num_queries` | Learnable queries used to score tokens. | `1` |
| `num_heads` | Cross-attention heads; must divide embedding width. | `1` |
| `pre_attn_norm` | Normalize tokens before cross-attention scoring. | `false` |
| `q_proj` | Learn a projection for queries. | `false` |
| `k_proj` / `v_proj` | Learn key/value projections. | `true` / `true` |
| `mlp` | Enable Cropr's feed-forward transformation. | `true` |
| `mlp_ratio` | Cropr feed-forward width divided by embedding width. | `4.0` |

The saved vision configuration uses `token_compression: cropr`, `cropr_cfg`, and `pruning_rate` for these settings. Optional `cropr_after_last_block` defaults to `false`, inserting Cropr after all repeated blocks except the last. With six `MMMT` blocks there are five Cropr modules. A pruning rate of `0.8` targets 20% remaining tokens overall, distributed across those modules; integer token counts are rounded.

## Logging and reporting

Both shell scripts write `full.log` under their run directory. Set a fresh `run_name` to distinguish experiments; reusing a directory can overwrite the console log and checkpoints.

| Stage | Argument | Meaning |
| --- | --- | --- |
| MAE | `--log_dir` | TensorBoard event directory; enabled by the launcher. |
| MAE | `--wandb` | Enable Weights & Biases reporting. |
| MAE | `--wandb_project` | W&B project; defaults to `mae-pretrain`. |
| MAE | `--wandb_run_name` | Optional W&B run name. |
| CLIP | `--report-to` | `wandb`, `tensorboard`, or `wandb,tensorboard`; disabled in the template. |
| CLIP | `--wandb-project-name` / `--wandb-run-name` | W&B project and run names. |
| CLIP | `--log-every-n-steps` | Console/reporting interval in training steps. Template: `1`. |

The complete CLI parsers, including inherited options outside this WSI workflow, are available through:

```bash
python -m slicechat_encoder.train.mae.main_pretrain --help
python -m slicechat_encoder.train.open_clip.open_clip_train.main --help
```
