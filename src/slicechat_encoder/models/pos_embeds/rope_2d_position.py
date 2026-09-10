from functools import partial

import torch
import torch.nn as nn
from einops import rearrange


def init_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs


def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    # t_x, t_y are (B, N)
    # freqs are (2, H, D_head/2)
    B, N = t_x.shape

    with torch.amp.autocast('cuda', enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(B, N, num_heads, -1).permute(0, 2, 1, 3) # B, H, N, D/2
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(B, N, num_heads, -1).permute(0, 2, 1, 3) # B, H, N, D/2
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    # freqs_cis is B,H,N,D/2. x is B*H,N,D/2
    # target shape for freqs_cis is B*H,N,D/2 to match x
    if freqs_cis.ndim == 4 and x.ndim == 3 and freqs_cis.shape[0] == x.shape[0] // freqs_cis.shape[1]:
        B, H, N, D_half = freqs_cis.shape
        return freqs_cis.reshape(B*H, N, D_half)

    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
    else:
        raise RuntimeError(f"Unexpected freqs_cis shape {freqs_cis.shape} for tensor shape {x.shape}")
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    #freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class RoPE2D(nn.Module):
    def __init__(self, args, head_dim, num_heads):
        super().__init__()
        self.args = args
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.rope_mixed = getattr(args, 'rope_2d_mixed', True)
        self.rope_theta = getattr(args, 'rope_2d_theta', 10.0)
        
        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)
            freqs = init_2d_freqs(
                dim=self.head_dim, num_heads=self.num_heads, theta=self.rope_theta, 
                rotate=True
            ).view(2, -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
        else:
            raise NotImplementedError("Axial RoPE is not yet supported in this integrated module.")

    def forward(self, q, k, positions):
        # positions: (B, N, 2)
        # q, k: (B*H, N, D)
        if self.rope_mixed:
            t_x = positions[..., 0] # B, N
            t_y = positions[..., 1] # B, N
            freqs_cis = self.compute_cis(self.freqs.float(), t_x.float(), t_y.float()) # TODO: Init this as float32
            freqs_cis = freqs_cis.to(q.device)
            q = q.view(freqs_cis.shape[0], freqs_cis.shape[1], -1, self.head_dim)
            k = k.view(freqs_cis.shape[0], freqs_cis.shape[1], -1, self.head_dim)
            xq_out, xk_out = apply_rotary_emb(q, k, freqs_cis)
            return rearrange(xq_out, 'b h n d -> (b h) n d'), rearrange(xk_out, 'b h n d -> (b h) n d')
        else:
            # Axial RoPE would be implemented here
            return q, k
