import torch


@torch.no_grad()
def _raster_scan_perm(
    positions: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute a per-batch left-to-right, top-to-bottom permutation.

    Returns:
      perm: [B, L] long
    """
    # Infer padding mask if not provided: pad if any coord == -1
    if pad_mask is None:
        pad_mask = (positions[..., 0] == -1) | (positions[..., 1] == -1)
        pad_mask = pad_mask.to(torch.bool)
    else:
        pad_mask = (pad_mask != 0)

    valid = ~pad_mask
    x = positions[..., 0]
    y = positions[..., 1]

    if x.dtype.is_floating_point:
        sentinel = torch.finfo(x.dtype).max
    else:
        sentinel = torch.iinfo(x.dtype).max

    xk = torch.where(valid, x, torch.full_like(x, sentinel))
    yk = torch.where(valid, y, torch.full_like(y, sentinel))

    # Stable lexicographic sort via two stable argsorts:
    # 1) sort by secondary key
    # 2) stable sort by primary key
    perm_secondary = torch.argsort(xk, dim=1, stable=True)  # secondary: x
    primary_sorted = yk.gather(1, perm_secondary)           # primary: y

    perm_primary = torch.argsort(primary_sorted, dim=1, stable=True)
    return perm_secondary.gather(1, perm_primary)


def raster_row_first_sort(
    positions: torch.Tensor,          # [B, L, 2], float/int, pads are -1
    embeds: torch.Tensor,             # [B, L, D], float
    pad_mask: torch.Tensor | None = None,  # [B, L], 1=pad, 0=valid  (optional)
    residual: torch.Tensor | None = None,  # [B, L, D], float (optional, sorted like embeds)
) -> dict:
    """
    Raster scan sort: left-to-right, top-to-bottom (sort by y, then x).

    Returns the same keys as hilbert_sort for interface compatibility.
    """
    assert positions.ndim == 3 and positions.shape[-1] == 2, "positions must be [B, L, 2]"
    assert embeds.shape[:2] == positions.shape[:2], "embeds and positions must share [B, L]"
    B, L, _ = positions.shape
    device = positions.device

    with torch.no_grad():
        perm = _raster_scan_perm(positions, pad_mask=pad_mask)

    inv_perm = torch.empty_like(perm)
    inv_perm.scatter_(1, perm, torch.arange(L, device=device).unsqueeze(0).expand(B, -1))

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

    # Compatibility field name with hilbert_sort:
    # assign each input token its final sorted rank.
    raster_rank = torch.empty_like(perm)
    raster_rank.scatter_(1, perm, torch.arange(L, device=device).unsqueeze(0).expand(B, -1))

    return {
        "positions_sorted": positions_sorted,
        "embeds_sorted": embeds_sorted,
        "residual_sorted": residual_sorted,
        "perm": perm,
        "inv_perm": inv_perm,
        "hilbert_keys": raster_rank,
    }
