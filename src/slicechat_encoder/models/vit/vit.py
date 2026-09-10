# Generic Vision Transformer block components with variable length flash attention support

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

from .config import ViTBlockConfig

# Prefer timm's trunc_normal_ for ViT-style init; fallback to PyTorch if unavailable.
try:
    from timm.layers.weight_init import trunc_normal_ as _timm_trunc_normal_
except Exception:
    _timm_trunc_normal_ = None

# Try to import flash attention
try:
    from .flash_attention import flash_attn_func, flash_attn_varlen_func
    HAS_FLASH_ATTN = flash_attn_varlen_func is not None
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_varlen_func = None
    flash_attn_func = None


# Activation functions mapping
ACT2FN = {
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
    "silu": F.silu,
    "swish": F.silu,
}


class ViTAttention(nn.Module):
    """Multi-headed self-attention for Vision Transformer with flash attention support."""

    def __init__(
        self,
        config: ViTBlockConfig,
        use_flash_attn: bool = True,
        qkv_bias: bool = False,
        rope_2d: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} "
                f"and `num_heads`: {self.num_heads})."
            )
        
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        self.use_flash_attn = use_flash_attn and HAS_FLASH_ATTN

        # DINO/timm-style default: qkv_bias=False
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        # Keep output projection bias (matches common ViT implementations)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Optional 2D Rotary Position Embedding
        self.rope_2d = rope_2d

    def _flash_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Flash attention forward pass.
        
        Args:
            queries: (batch, seq_len, num_heads, head_dim)
            keys: (batch, seq_len, num_heads, head_dim)
            values: (batch, seq_len, num_heads, head_dim)
            key_padding_mask: (batch, seq_len) - True for valid tokens, False for padding
            
        Returns:
            Output tensor of shape (batch, seq_len, num_heads, head_dim)
        """
        dropout = self.dropout if self.training else 0.0
        
        if key_padding_mask is not None:
            # Use variable length flash attention for padded sequences
            attn_output, _ = flash_attn_varlen_func(
                queries, keys, values,
                dropout=dropout,
                bias=None,
                key_padding_mask=key_padding_mask,
                softmax_scale=self.scale,
                is_causal=False,
            )
        else:
            # Use standard flash attention
            attn_output, _ = flash_attn_func(
                queries, keys, values,
                dropout=dropout,
                bias=None,
                softmax_scale=self.scale,
                is_causal=False,
            )
        
        return attn_output

    def _standard_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Standard scaled dot-product attention.
        
        Args:
            queries: (batch, num_heads, seq_len, head_dim)
            keys: (batch, num_heads, seq_len, head_dim)
            values: (batch, num_heads, seq_len, head_dim)
            key_padding_mask: (batch, seq_len) - True for valid tokens, False for padding
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Scaled dot-product attention
        attn_weights = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            # Convert boolean mask to attention mask
            # key_padding_mask: (batch, seq_len), True = valid, False = padding
            # We need to mask out padding positions in attention
            # Expand to (batch, 1, 1, seq_len) for broadcasting
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(~attn_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, values)
        
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        pooler: Optional[str] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, embed_dim)
            key_padding_mask: Boolean mask of shape (batch_size, seq_length)
                              True for valid tokens, False for padding tokens
            positions: Optional (batch_size, num_patches, 2) coordinates for RoPE 2D.
                       Only used when rope_2d is set. Does NOT include the CLS token.
            pooler: Optional pooler type string (e.g. "cls"). When "cls", the first
                    token is excluded from RoPE application.
            
        Returns:
            Tuple of (output tensor, attention weights or None for flash attention)
        """
        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Apply 2D Rotary Position Embedding to q, k
        if self.rope_2d is not None and positions is not None:
            q_rope = rearrange(queries, 'b n (h d) -> (b h) n d', h=self.num_heads)
            k_rope = rearrange(keys, 'b n (h d) -> (b h) n d', h=self.num_heads)

            if pooler == "cls":
                # Skip the CLS token (first position) for RoPE
                q_r, k_r = self.rope_2d(q_rope[:, 1:, :], k_rope[:, 1:, :], positions)
                q_rope = torch.cat((q_rope[:, :1, :], q_r), dim=1)
                k_rope = torch.cat((k_rope[:, :1, :], k_r), dim=1)
            else:
                q_rope, k_rope = self.rope_2d(q_rope, k_rope, positions)

            queries = rearrange(q_rope, '(b h) n d -> b n (h d)', h=self.num_heads)
            keys = rearrange(k_rope, '(b h) n d -> b n (h d)', h=self.num_heads)

        if self.use_flash_attn:
            # Flash attention expects (batch, seq_len, num_heads, head_dim)
            queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim)
            keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim)
            values = values.view(batch_size, seq_length, self.num_heads, self.head_dim)
            
            attn_output = self._flash_attention(queries, keys, values, key_padding_mask)
            attn_output = attn_output.reshape(batch_size, seq_length, embed_dim)
            attn_output = self.out_proj(attn_output)
            
            return attn_output, None
        else:
            # Standard attention expects (batch, num_heads, seq_len, head_dim)
            queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

            attn_output, attn_weights = self._standard_attention(queries, keys, values, key_padding_mask)

            # Reshape back to (batch, seq_len, embed_dim)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(batch_size, seq_length, embed_dim)
            
            attn_output = self.out_proj(attn_output)

            return attn_output, attn_weights


class ViTMLP(nn.Module):
    """MLP block for Vision Transformer."""
    
    def __init__(self, config: ViTBlockConfig):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ViTBlock(nn.Module):
    """Single Vision Transformer block with pre-norm architecture."""
    
    def __init__(
        self,
        config: ViTBlockConfig,
        drop_path_rate: float = 0.0,
        use_flash_attn: bool = True,
        init_weights: bool = True,
        rope_2d: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = ViTAttention(
            config,
            use_flash_attn=use_flash_attn,
            qkv_bias=getattr(config, "qkv_bias", False),
            rope_2d=rope_2d,
        )
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = ViTMLP(config)
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        if init_weights:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """
        timm-style initialization:
        - Linear weights: truncated normal (std=0.02)
        - Linear biases: zeros
        - LayerNorm: weight=1, bias=0
        """
        if isinstance(m, nn.Linear):
            if _timm_trunc_normal_ is not None:
                _timm_trunc_normal_(m.weight, std=0.02)
            else:
                nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        pooler: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, embed_dim)
            key_padding_mask: Boolean mask of shape (batch_size, seq_length)
                              True for valid tokens, False for padding tokens
            positions: Optional (batch_size, num_patches, 2) coordinates for RoPE 2D.
            pooler: Optional pooler type string. Used to determine if the CLS token should be excluded from RoPE application. (e.g. "cls").
            
        Returns:
            Output tensor of shape (batch_size, seq_length, embed_dim)
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, key_padding_mask, positions=positions, pooler=pooler)
        hidden_states = residual + self.drop_path1(hidden_states)

        # MLP with residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.drop_path2(hidden_states)

        return hidden_states
