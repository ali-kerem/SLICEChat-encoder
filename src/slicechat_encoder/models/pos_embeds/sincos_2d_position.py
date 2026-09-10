import torch


def _get_1d_sincos_pos_embed_from_coords(embed_dim: int, pos: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=dtype)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    pos = pos.reshape(-1).to(dtype=dtype)  # (n,)
    out = pos[:, None] * omega[None, :]  # (n, D/2)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)  # (n, D)


def get_sincos_2d_pos_embed_from_coords(coords: torch.Tensor, embed_dim: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    assert embed_dim % 2 == 0, "embed_dim must be even"
    x, y = coords[:, 0], coords[:, 1]  # (n,), (n,)
    emb_x = _get_1d_sincos_pos_embed_from_coords(embed_dim // 2, x, dtype=dtype)
    emb_y = _get_1d_sincos_pos_embed_from_coords(embed_dim // 2, y, dtype=dtype)
    return torch.cat([emb_x, emb_y], dim=1)  # (n, embed_dim)
