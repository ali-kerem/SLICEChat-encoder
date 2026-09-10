import math
from functools import partial
import torch
import torch.nn as nn
from einops import rearrange

from timm.layers.weight_init import trunc_normal_tf_
from timm.layers.mlp import Mlp

from ..sorters import hilbert_sort


class CrossAttention(nn.Module):
    """Cross-attention layer with learnable queries."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        pre_attn_norm: bool = False,
        q_proj: bool = False,
        k_proj: bool = False,
        v_proj: bool = False,
        mlp: bool = True,
        num_queries: int = 1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.queries = nn.Parameter(torch.empty(1, num_queries, embed_dim))

        self.q = (
            nn.Linear(embed_dim, embed_dim, bias=qkv_bias) if q_proj else nn.Identity()
        )
        self.k = (
            nn.Linear(embed_dim, embed_dim, bias=qkv_bias) if k_proj else nn.Identity()
        )
        self.attn_norm = norm_layer(embed_dim) if pre_attn_norm else nn.Identity()

        self.v = (
            nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
            if v_proj
            else nn.Identity()
        )
        self.proj = (
            nn.Linear(embed_dim, embed_dim) if num_heads > 1 else nn.Identity()
        )

        if mlp:
            self.mlp_norm = norm_layer(embed_dim)
            self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio))

        self.init_weights()

    def init_weights(self):
        D = self.queries.shape[-1]
        trunc_normal_tf_(self.queries, std=D**-0.5)

    def forward_scorer(self, x, mask=None):
        B, N = x.shape[:2]

        x = self.attn_norm(x) # B, N, D

        q = self.q(self.queries).reshape(1, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2).expand(B, -1, -1, -1) # B, H, num_queries, D/H
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) # B, H, N, D/H

        mask = mask[:, None, None, :].expand(B, self.num_heads, q.shape[2], N) # B, H, num_queries, N

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale # B, H, num_queries, total_valid_tokens
        attn[mask == 0] = -math.inf
        scores = attn.sum((1, 2))  # B, N
        return scores

    def forward(self, x, mask=None):
        B, N, D = x.shape # B, N, D

        x = self.attn_norm(x)

        q = self.q(self.queries).reshape(1, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2).expand(B, -1, -1, -1) # B, H, num_queries, D/H
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) # B, H, N, D/H
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) # B, H, N, D/H


        mask = mask[:, None, None, :].expand(B, self.num_heads, q.shape[2], N) # B, H, num_queries, N

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale # B, H, num_queries, total_valid_tokens
        attn[mask == 0] = -math.inf
        scores = attn.sum((1, 2))  # B, N : scores from pre-softmax logits
        attn = attn.float().softmax(-1).to(attn.dtype)  # softmax in float32, then cast back
        out = torch.matmul(attn, v) # B, num_heads, num_queries, D/head
        out = rearrange(out, 'b h q d -> b q (h d)') # B, num_queries, D : concat heads

        out = self.proj(out) # B, num_queries, D

        if hasattr(self, "mlp"):
            out = out + self.mlp(self.mlp_norm(out))

        out = out.mean(dim=1) # B, D : average over queries

        return out, scores


class Cropr(nn.Module):
    def __init__(
        self,
        keep_rate: float = 0.75,
        num_queries: int = 1,
        embed_dim: int = 768,
        num_heads: int = 16,
        pre_attn_norm: bool = False,
        q_proj: bool = False,
        k_proj: bool = False,
        v_proj: bool = False,
        mlp: bool = True,
        mlp_ratio: float = 4.0,
        pooler: str = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.keep_rate = keep_rate
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.mlp = mlp
        self.pooler = pooler
        self.sorter = hilbert_sort

        self.cross_attn = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pre_attn_norm=pre_attn_norm,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            mlp=mlp,
            num_queries=num_queries,
        )

    def prune(self, x, scores, mask, positions, residual=None):
        num_tokens = mask.sum(-1) # B, N -> B
        num_keep = torch.round(num_tokens * self.keep_rate).long() # B

        B, _, D = x.shape
        segment_lengths = num_tokens - num_keep # B: contiguous segment size to remove
        max_keep = int(num_keep.max().item())
        is_cls = self.pooler == "cls"
        pos_dim = positions.shape[-1]

        x_out = x.new_zeros(B, max_keep, D)
        pos_out = positions.new_zeros(B, max_keep - int(is_cls), pos_dim)
        residual_out = None
        if residual is not None:
            residual_out = residual.new_zeros(B, max_keep, residual.shape[-1])

        for b in range(B):
            L = int(num_tokens[b].item())
            K = int(num_keep[b].item())
            S = int(segment_lengths[b].item())

            if S <= 0:
                x_out[b, :K] = x[b, :K]
                pos_out[b, :K - int(is_cls)] = positions[b, :K - int(is_cls)]
                if residual is not None:
                    residual_out[b, :K] = residual[b, :K]
                continue

            # When CLS is present, search only over feature tokens (skip index 0)
            if is_cls:
                feat_scores = scores[b, 1:L]
                n_feat = L - 1
            else:
                feat_scores = scores[b, :L]
                n_feat = L

            # Sliding window sums of size S via cumulative sum (float32 to avoid bfloat16 cumsum drift)
            cs = torch.cumsum(feat_scores.float(), dim=0)
            cs = torch.cat([cs.new_zeros(1), cs])
            n_windows = n_feat - S + 1
            window_sums = cs[S:S + n_windows] - cs[:n_windows]

            # Remove the segment with the lowest total score
            best_feat_start = torch.argmin(window_sums).item()
            best_start = best_feat_start + (1 if is_cls else 0)

            # Keep tokens before and after the pruned segment (original order preserved)
            keep_idx = torch.cat([
                torch.arange(best_start, device=x.device),
                torch.arange(best_start + S, L, device=x.device)
            ])

            x_out[b, :K] = x[b, keep_idx]
            if is_cls:
                pos_keep_idx = keep_idx[1:] - 1
                pos_out[b, :K - 1] = positions[b, pos_keep_idx]
            else:
                pos_out[b, :K] = positions[b, keep_idx]

            if residual is not None:
                residual_out[b, :K] = residual[b, keep_idx]

        indices = torch.arange(max_keep, device=x.device)
        output_mask = (indices < num_keep.unsqueeze(1)).long()

        return x_out, output_mask, pos_out, residual_out

    def _apply_prune_ordering(self, x, mask, positions, residual=None):
        if positions is None:
            raise ValueError("Cropr `_apply_prune_ordering` requires positions.")

        is_cls = self.pooler == "cls"
        if is_cls:
            feat_x = x[:, 1:]
            pad_mask = 1 - mask[:, 1:]
            feat_residual = residual[:, 1:] if residual is not None else None
        else:
            feat_x = x
            pad_mask = 1 - mask
            feat_residual = residual

        sorter_out = self.sorter(
            positions,
            feat_x,
            pad_mask=pad_mask,
            residual=feat_residual,
        )
        feat_x = sorter_out["embeds_sorted"]
        positions = sorter_out["positions_sorted"]
        feat_residual = sorter_out["residual_sorted"] if residual is not None else None

        if is_cls:
            x = torch.cat([x[:, :1], feat_x], dim=1)
            if residual is not None:
                residual = torch.cat([residual[:, :1], feat_residual], dim=1)
        else:
            x = feat_x
            residual = feat_residual

        return x, mask, positions, residual

    def forward(self, x, mask=None, positions=None, residual=None):
        mask = 1 - mask # Turn padding mask into validity mask
        x, mask, positions, residual = self._apply_prune_ordering(
            x, mask, positions, residual=residual
        )
        if not self.training:
            scores = self.cross_attn.forward_scorer(x, mask=mask)
        else:
            # use detach to stop gradients from flowing back
            pred, scores = self.cross_attn(x.detach(), mask=mask)

        if self.pooler == "cls":
            scores[:, 0] = math.inf  # retain CLS token
        scores[mask == 0] = -math.inf # Discard padding tokens

        x, new_mask, new_positions, residual = self.prune(
            x, scores, mask, positions, residual=residual
        )
        new_mask = 1 - new_mask # Turn validity mask into padding mask

        if not self.training:
            return x, None, new_mask, new_positions, residual

        return x, pred, new_mask, new_positions, residual
