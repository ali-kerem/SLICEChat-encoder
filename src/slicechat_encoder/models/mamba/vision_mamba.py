# Vision Mamba encoder for pre-extracted WSI features
# Based on MAP (Mamba Autoregressive Pre-training) architecture
# Reference: https://github.com/yunzeliu/MAP

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from timm.models.layers import DropPath
from ..vit.vit import ViTBlock

from ..pos_embeds import get_sincos_2d_pos_embed_from_coords, RoPE2D
from ..poolers.factory import PoolerFactory
from ..token_compressors import Cropr
from ..sorters import raster_row_first_sort

from .mamba_custom import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def _init_mamba_gpt2_style_weights(
    module: nn.Module,
    n_layer: int,
    initializer_range: float = 0.02,  # Only used for nn.Embedding, kept for parity with MAP/ViM.
    rescale_prenorm_residual: bool = True,
    n_residuals_per_layer: int = 1,
):
    """
    MAP/ViM-style initialization helper intended to be used via `model.apply(partial(...))`.

    Key properties:
    - Respects special parameters marked with `_no_reinit` (e.g., Mamba dt_proj.bias).
    - Does NOT blanket re-initialize all Linear weights (avoids clobbering Mamba's custom init).
    - Optionally applies scaled re-init to selected residual projection weights.
    """
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Re-init + scale selected residual projection weights (GPT-2 style).
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MambaBlock(nn.Module):
    """Single Mamba block with residual connection and normalization."""
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        drop_path: float = 0.0,
        rms_norm: bool = False,
        norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = False,
        fused_add_norm: bool = False,
        biscan: bool = True,
        layer_idx: int = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        
        # Mamba mixer
        if Mamba is None:
            raise ImportError("mamba_ssm is required. Install with: pip install mamba-ssm")
        
        self.mixer = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            biscan=biscan,
            layer_idx=layer_idx,
            **factory_kwargs,
        )
    
        
        # Normalization
        norm_cls = RMSNorm if (rms_norm and RMSNorm is not None) else nn.LayerNorm
        self.norm = norm_cls(dim, eps=norm_epsilon, **factory_kwargs)
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm))
    
    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None, # NOTE: new
        inference_params=None,
    ):
        """
        Args:
            hidden_states: (B, L, D) input tensor
            residual: Optional residual for fused add norm
            inference_params: Optional inference parameters
            
        Returns:
            hidden_states: (B, L, D) output tensor
            residual: Residual for next layer
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        
        hidden_states = self.mixer(
            hidden_states,
            inference_params=inference_params,
            padding_mask=padding_mask, # NOTE: new
        )
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class VisionMambaEncoder(nn.Module):
    """
    Vision Mamba encoder for pre-extracted WSI features.
    
    Unlike standard Vision Mamba which operates on images:
    - No patch embedding: features are already extracted (e.g., UNI, CONCH, Virchow)
    - Uses 2D sincos positional embeddings from coordinates
    - Handles variable-length sequences with padding
    - Supports different pooling strategies
    - Optionally interleaves transformer attention blocks (hybrid)
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        embed_dim = args.encoder_embed_dim
        block_structure = args.block_structure
        num_blocks = int(args.num_blocks)
        encoder_layout = tuple(block_structure * num_blocks)
        num_mamba_layers = encoder_layout.count("M")
        num_hybrid_blocks = encoder_layout.count("T")
        self.num_mamba_layers = num_mamba_layers
        self.num_hybrid_blocks = num_hybrid_blocks
        self.block_structure = block_structure
        self.num_blocks = num_blocks
        self.encoder_layout = encoder_layout
        block_len = len(block_structure)
        block_end_indices = [((idx + 1) * block_len) - 1 for idx in range(num_blocks)]
        self.block_end_indices = set(block_end_indices)
        
        # Store parameters
        self.embed_dim = embed_dim
        self.residual_in_fp32 = args.residual_in_fp32
        self.fused_add_norm = args.fused_add_norm
        
        # Positional embeddings
        if args.pos_embed == "sincos_2d":
            self.embed_positions = get_sincos_2d_pos_embed_from_coords
        else:
            self.embed_positions = None
        
        # Dropout
        self.dropout_module = nn.Dropout(args.dropout)
        
        # Pooler
        self.pooler = PoolerFactory.create(
            pooler_type=args.pooler,
            hidden_size=embed_dim,
            intermediate_size=embed_dim * 4,
            num_heads=args.hybrid_vit_config.num_attention_heads,
        )
        
        # CLS token for cls pooling
        if args.pooler == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)
        
        # Drop path rate
        dpr = [x.item() for x in torch.linspace(0, args.drop_path_rate, self.num_mamba_layers)]
        inter_dpr = [0.0] + dpr
        
        # Mamba layers
        self.layers = nn.ModuleList([
            MambaBlock(
                dim=embed_dim,
                d_state=args.d_state,
                d_conv=args.d_conv,
                expand=args.expand,
                drop_path=inter_dpr[i],
                rms_norm=args.rms_norm,
                norm_epsilon=args.layernorm_eps,
                residual_in_fp32=args.residual_in_fp32,
                fused_add_norm=args.fused_add_norm,
                biscan=args.biscan,
                layer_idx=i,
            )
            for i in range(self.num_mamba_layers)
        ])

        # Hybrid transformer blocks (interleaved attention)
        if num_hybrid_blocks > 0:
            if args.hybrid_vit_config.hidden_size != embed_dim:
                raise ValueError(
                    f"`hybrid_vit_config.hidden_size` must match encoder embed dim "
                    f"(got {args.hybrid_vit_config.hidden_size} vs encoder_embed_dim={embed_dim})."
                )

            head_dim = embed_dim // args.hybrid_vit_config.num_attention_heads
            self.hybrid_blocks = nn.ModuleList([
                ViTBlock(
                    config=args.hybrid_vit_config,
                    drop_path_rate=0.0,
                    use_flash_attn=True,
                    init_weights=True,
                    rope_2d=RoPE2D(args, head_dim, args.hybrid_vit_config.num_attention_heads) if args.pos_embed == "rope_2d" else None,
                )
                for _ in range(num_hybrid_blocks)
            ])
        else:
            self.hybrid_blocks = None
        
        # Final normalization
        norm_cls = RMSNorm if (args.rms_norm and RMSNorm is not None) else nn.LayerNorm
        if args.normalize_output:
            self.norm_f = norm_cls(embed_dim, eps=args.layernorm_eps)
        else:
            self.norm_f = None
        
        # Drop path for final residual
        self.drop_path = DropPath(args.drop_path_rate) if args.drop_path_rate > 0. else nn.Identity()

        self.sorter = raster_row_first_sort

        if args.token_compression == "cropr":
            num_cropr_blocks = num_blocks if args.cropr_after_last_block else max(num_blocks - 1, 0)
            if num_cropr_blocks < 1:
                raise ValueError(
                    "Cropr requires at least one block boundary before the final output. "
                    "Use num_blocks >= 2, or set cropr_after_last_block=True to compress "
                    "after the final block."
                )
            self.cropr_block_end_indices = set(block_end_indices[:num_cropr_blocks])
            per_layer_keep_rate = (1 - args.pruning_rate) ** (1 / num_cropr_blocks)
            self.cropr = nn.ModuleList(
                [
                    Cropr(
                        embed_dim=embed_dim,
                        keep_rate=per_layer_keep_rate,
                        pooler=args.pooler,
                        **args.cropr_cfg
                    )
                    for _ in range(num_cropr_blocks)
                ]
            )
        else:
            self.cropr_block_end_indices = set()
        
        # Initialize weights (safe MAP/ViM style; does not clobber Mamba special init) # NOTE: different from MAP
        self.apply(partial(_init_mamba_gpt2_style_weights, n_layer=max(self.num_mamba_layers, 1)))

    @staticmethod
    def _materialize_stream(hidden_states: Tensor, residual: Optional[Tensor]) -> Tensor:
        if residual is None:
            return hidden_states
        return (residual + hidden_states).to(dtype=hidden_states.dtype)
    
    def forward_embedding(self, token_embeddings: Tensor, positions: Tensor):
        """Apply positional embeddings to token embeddings."""
        x = token_embeddings
        
        if self.embed_positions is not None and positions is not None:
            batch_size, num_patches, embed_dim = x.shape
            
            # Reshape coords from (B, N, 2) to (B*N, 2)
            coords_reshaped = positions.reshape(-1, 2)
            
            # Get positional embeddings
            pos_embed = self.embed_positions(coords_reshaped, embed_dim, dtype=x.dtype)
            pos_embed = pos_embed.reshape(batch_size, num_patches, embed_dim)
            x = x + pos_embed
        
        x = self.dropout_module(x)
        return x
    
    def sort_tokens(self, x, positions, pad_mask, *, return_inv_perm: bool = False, residual=None):
        """
        Sort tokens using the configured sorter.

        If return_inv_perm=True, also returns an inverse permutation (inv_perm_full) such that:
          x_restored = x_sorted.gather(1, inv_perm_full[..., None].expand(-1, -1, D))
        restores x_sorted back to the *input* token order.

        If residual is provided it is reordered with the same permutation as x.
        """
        inv_perm_full = None
        residual_sorted = None
        if self.args.pooler == "cls":
            res_slice = residual[:, 1:] if residual is not None else None
            out = self.sorter(positions[:, 1:], x[:, 1:], pad_mask=pad_mask[:, 1:], residual=res_slice)
            x = torch.cat([x[:, :1], out["embeds_sorted"]], dim=1)
            positions = torch.cat([positions[:, :1], out["positions_sorted"]], dim=1)
            if residual is not None:
                residual_sorted = torch.cat([residual[:, :1], out["residual_sorted"]], dim=1)
            if return_inv_perm:
                inv_perm_sub = out["inv_perm"]  # [B, L-1]
                inv_perm_full = torch.empty(
                    (inv_perm_sub.shape[0], inv_perm_sub.shape[1] + 1),
                    device=inv_perm_sub.device,
                    dtype=inv_perm_sub.dtype,
                )
                inv_perm_full[:, 0] = 0
                inv_perm_full[:, 1:] = inv_perm_sub + 1
        else:
            out = self.sorter(positions, x, pad_mask=pad_mask, residual=residual)
            x = out["embeds_sorted"]
            positions = out["positions_sorted"]
            residual_sorted = out["residual_sorted"]
            if return_inv_perm:
                inv_perm_full = out["inv_perm"]

        return x, positions, inv_perm_full, residual_sorted

    def forward(
        self,
        token_embeddings: Tensor,
        positions: Tensor,
        encoder_padding_mask: Tensor,
        return_pooled_output: bool = True,
        retain_order: bool = False,
    ):
        """
        Forward pass through Vision Mamba encoder.
        
        Args:
            token_embeddings: (B, L, D) input features
            positions: (B, L, 2) coordinates for positional embeddings
            encoder_padding_mask: (B, L) padding mask (1=padded, 0=valid)
            return_pooled_output: Whether to return pooled output
            
        Returns:
            dict with:
                - encoder_out: (B, L, D) encoded features
                - padding_mask: (B, L) padding mask
                - pooled_out: (B, D) pooled output (if return_pooled_output=True)
        """
        encoder_padding_mask = encoder_padding_mask.to(dtype=torch.int64)
        
        # Normalize input if configured
        if self.args.normalize_image_patches:
            token_embeddings = F.normalize(token_embeddings, dim=-1)
        
        # Apply positional embeddings
        x = self.forward_embedding(token_embeddings, positions)
        
        # Prepend CLS token if using cls pooler
        if self.args.pooler == "cls":
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # Update padding mask
            cls_padding_mask = torch.zeros((batch_size, 1), dtype=torch.int64, device=x.device)
            encoder_padding_mask = torch.cat([cls_padding_mask, encoder_padding_mask], dim=1)
        
        # Zero out padded positions
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # Sort tokens before entering layers
        if self.sorter is not None:
            if retain_order:
                # Retaining the original order is only well-defined if sequence length is preserved.
                assert self.args.token_compression is None, (
                    "retain_order=True is incompatible with token compression "
                    f"(got token_compression={self.args.token_compression!r})."
                )
            x, positions, inv_perm, _ = self.sort_tokens(
                x, positions, encoder_padding_mask, return_inv_perm=True
            )

        # Forward through the configured block layout.
        residual = None
        hidden_states = x

        cropr_preds = []
        mamba_idx = 0
        transformer_idx = 0
        cropr_idx = 0
        for op_idx, op in enumerate(self.encoder_layout):
            if op == "M":
                hidden_states, residual = self.layers[mamba_idx](
                    hidden_states,
                    residual,
                    padding_mask=encoder_padding_mask, # NOTE: new
                )
                mamba_idx += 1
            elif op == "T":
                # ViTBlock expects key_padding_mask: True = valid tokens, False = padding.
                key_padding_mask = (encoder_padding_mask == 0)
                # Run Transformer on the materialized token stream and reset bookkeeping
                # so following Mamba/Transformer blocks do not add the same stream again.
                x_stream = self._materialize_stream(hidden_states, residual)
                x_stream = self.hybrid_blocks[transformer_idx](
                    x_stream,
                    key_padding_mask=key_padding_mask,
                    positions=positions,
                    pooler=self.args.pooler,
                )
                transformer_idx += 1
                residual = x_stream
                hidden_states = torch.zeros_like(x_stream)
            else:
                raise RuntimeError(f"Unexpected encoder op {op!r}.")

            if (
                self.args.token_compression == "cropr"
                and op_idx in self.cropr_block_end_indices
            ):
                # CroPr needs the concrete token stream. If a block ends in Mamba,
                # materialize residual + hidden_states before pruning, then reset
                # the two-stream Mamba bookkeeping exactly like a Transformer boundary.
                x_stream = self._materialize_stream(hidden_states, residual)
                x_stream, pred, encoder_padding_mask, positions, residual = self.cropr[cropr_idx](
                    x_stream,
                    mask=encoder_padding_mask,
                    positions=positions,
                    residual=x_stream,
                )
                cropr_idx += 1
                if pred is not None:
                    cropr_preds.append(pred)

                if residual is None:
                    residual = x_stream
                hidden_states = torch.zeros_like(residual)

                if self.sorter is not None:
                    x_stream, positions, _, residual = self.sort_tokens(
                        x_stream, positions, encoder_padding_mask,
                        residual=residual,
                    )
                    if residual is None:
                        residual = x_stream
                    hidden_states = torch.zeros_like(residual)
            
        # Final layer norm
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype)) if self.norm_f else residual
        else:
            # `fused_add_norm` requires a norm module. If output norm is disabled, fall back
            # to the unfused residual combine (keeps forward functional instead of crashing).
            if self.norm_f is None:
                hidden_states = hidden_states if residual is None else (residual + self.drop_path(hidden_states))
            else:
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                hidden_states = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
        
        x = hidden_states

        if retain_order and self.sorter is not None:
            D = x.shape[-1]
            x = x.gather(1, inv_perm.unsqueeze(-1).expand(-1, -1, D))
        
        # Build return dict
        return_dict = {
            "encoder_out": x,
            "padding_mask": encoder_padding_mask,
            "positions": positions,
        }
        
        if return_pooled_output:
            return_dict['pooled_out'] = self.pooler(x, padding_mask=encoder_padding_mask)

        if self.args.token_compression == "cropr":
            return_dict['cropr_preds'] = cropr_preds
        
        return return_dict


def make_mamba_encoder_from_config(config: dict):
    """Create VisionMambaEncoder from config dict."""
    from .config import MambaEncoderConfig
    
    config_obj = MambaEncoderConfig(**config)
    model = VisionMambaEncoder(config_obj)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable Mamba encoder parameters: {n_params:,}')
    
    return model
