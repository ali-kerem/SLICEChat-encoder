import torch
import math


@torch.no_grad()
def _hilbert_index_2d_batched(x: torch.Tensor, y: torch.Tensor, bits: int | None = None) -> torch.Tensor:
    """
    Vectorized 2D Hilbert index for batched sequences.
    x,y: (B,L) int64, >=0 for valid tokens. (Padded positions should be set to 0 beforehand.)
    bits: shared bit depth across the whole batch (chosen automatically if None).
    Returns: (B,L) int64 Hilbert index.
    """
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    B, L = x.shape
    max_xy = torch.maximum(x.max(), y.max())
    if bits is None:
        # bit_length exists on Python ints; convert safely
        bits = int(max_xy.item().bit_length()) if max_xy.numel() else 1
    if bits <= 0:
        return torch.zeros_like(x)

    # Keep n < 2^62 to avoid int64 overflow inside algorithm
    if bits >= 62:
        shift = bits - 61
        x = x >> shift
        y = y >> shift
        bits = 61

    n = 1 << bits
    d = torch.zeros_like(x, dtype=torch.int64)

    xi = x.clone()
    yi = y.clone()

    s = n >> 1
    while s > 0:
        rx = ((xi & s) != 0).to(torch.int64)
        ry = ((yi & s) != 0).to(torch.int64)

        d = d + (s * s) * ((3 * rx) ^ ry)

        mask = (ry == 0)
        rx1 = mask & (rx == 1)
        if rx1.any():
            xi = torch.where(rx1, (n - 1) - xi, xi)
            yi = torch.where(rx1, (n - 1) - yi, yi)

        if mask.any():
            xi_old = xi
            yi_old = yi
            xi = torch.where(mask, yi_old, xi_old)
            yi = torch.where(mask, xi_old, yi_old)

        s >>= 1

    return d


def hilbert_sort(
    positions: torch.Tensor,          # [B, L, 2], int (>=0), pads are -1
    embeds: torch.Tensor,             # [B, L, D], float
    pad_mask: torch.Tensor | None = None,  # [B, L], 1=pad, 0=valid  (optional)
    residual: torch.Tensor | None = None,  # [B, L, D], float (optional, sorted like embeds)
) -> dict:
    """
    Reorder positions and embeds per batch element by 2D Hilbert index.
    Padded tokens remain at the end of the sequence.

    If `residual` is provided it is reordered with the same permutation as embeds.

    Returns dict with:
      positions_sorted: [B, L, 2]
      embeds_sorted:    [B, L, D]
      residual_sorted:  [B, L, D] or None (present only when residual is given)
      perm:             [B, L] long (per-example permutation)
      inv_perm:         [B, L] long (inverse permutation)
      hilbert_keys:     [B, L] long (for inspection/debug)
    """
    assert positions.ndim == 3 and positions.shape[-1] == 2, "positions must be [B, L, 2]"
    assert embeds.shape[:2] == positions.shape[:2], "embeds and positions must share [B, L]"
    B, L, _ = positions.shape
    device = positions.device

    # Infer padding mask if not provided: pad if any coord == -1
    if pad_mask is None:
        pad_mask = (positions[..., 0] == -1) | (positions[..., 1] == -1)
        pad_mask = pad_mask.to(torch.bool)
    else:
        pad_mask = (pad_mask != 0)  # ensure bool

    # Valid tokens mask
    valid = ~pad_mask

    # Prepare x,y (set padded coords to 0 so Hilbert math is safe)
    x = positions[..., 0].clone().to(torch.int64)
    y = positions[..., 1].clone().to(torch.int64)
    x = torch.where(valid, x, torch.zeros_like(x))
    y = torch.where(valid, y, torch.zeros_like(y))

    # Choose a shared bit depth across the WHOLE batch (fully vectorized).
    # You can change this to per-example if you prefer slightly tighter packing per row,
    # but global bits is simpler and robust.
    # with torch.no_grad():
        # Only consider valid coords for bit depth
        # max_x = torch.where(valid, x, torch.zeros_like(x)).max()
        # max_y = torch.where(valid, y, torch.zeros_like(y)).max()
        # global_max = torch.maximum(max_x, max_y)
        # bits = int(global_max.item().bit_length()) if global_max > 0 else 1
    bits = 10

    # Compute Hilbert keys
    with torch.no_grad():
        keys = _hilbert_index_2d_batched(x, y, bits=bits)
        # Push padded tokens to the end using a large sentinel
        sentinel = (1 << 62) - 1
        keys = torch.where(valid, keys, torch.full_like(keys, sentinel))

    # Per-row argsort by keys
    perm = torch.argsort(keys, dim=1, stable=True)
    # Inverse permutation
    inv_perm = torch.empty_like(perm)
    inv_perm.scatter_(1, perm, torch.arange(L, device=device).unsqueeze(0).expand(B, -1))

    # Reorder tensors (gather keeps gradient flow for embeds)
    # positions are integer so grad isn't relevant; embeds keep full autograd.
    gather_idx_pos = perm.unsqueeze(-1).expand(-1, -1, 2)
    positions_sorted = positions.gather(1, gather_idx_pos)

    D = embeds.shape[-1]
    gather_idx_emb = perm.unsqueeze(-1).expand(-1, -1, D)
    embeds_sorted = embeds.gather(1, gather_idx_emb)

    residual_sorted = None
    if residual is not None:
        D_r = residual.shape[-1]
        gather_idx_res = perm.unsqueeze(-1).expand(-1, -1, D_r)
        residual_sorted = residual.gather(1, gather_idx_res)

    return {
        "positions_sorted": positions_sorted,
        "embeds_sorted": embeds_sorted,
        "residual_sorted": residual_sorted,
        "perm": perm,
        "inv_perm": inv_perm,
        "hilbert_keys": keys,
    }

