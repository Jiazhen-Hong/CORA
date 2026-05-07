import torch


def random_channel_mask(
    x: torch.Tensor,
    channel_mask_ratio: float,
    mode: str = "bernoulli",
    min_keep: int = 1,
    coords: torch.Tensor | None = None,
    group_frac: float = 0.25,
):
    B, C, _ = x.shape
    device, dtype = x.device, x.dtype
    ratio = float(max(0.0, min(1.0, channel_mask_ratio)))
    keep_prob = 1.0 - ratio

    if ratio <= 0:
        return torch.ones((B, C, 1), device=device, dtype=dtype)

    if mode == "bernoulli" or (coords is None):
        keep = torch.bernoulli(torch.full((B, C), keep_prob, device=device)).to(dtype)
        need = (keep.sum(dim=1) < min_keep)
        if need.any():
            for b in torch.where(need)[0].tolist():
                idx = torch.randperm(C, device=device)[:min_keep]
                keep[b, idx] = 1.0
        return keep.unsqueeze(-1)

    assert coords is not None and coords.dim() == 2 and coords.size(0) == C
    k = max(min_keep, int(round(C * group_frac)))
    k = max(1, min(C, k))
    keep = torch.ones((B, C), device=device, dtype=dtype)
    for b in range(B):
        center = torch.randint(0, C, (1,), device=device).item()
        d2 = ((coords - coords[center]) ** 2).sum(dim=1)
        nn_idx = torch.topk(-d2, k=k).indices
        m = int(max(1, round(ratio * C)))
        m = min(m, k)
        mask_idx = nn_idx[:m]
        keep[b, mask_idx] = 0.0
        if int(keep[b].sum().item()) < min_keep:
            add_idx = torch.randperm(C, device=device)[:min_keep]
            keep[b, add_idx] = 1.0
    return keep.unsqueeze(-1)


def apply_mask_ssp(
    input_tensor: torch.Tensor,
    time_mask_ratio: float = 0.5,
    num_preserved_blocks: int = 8,
    evenly_spaced: bool = False,
    min_block_len: int = 4,
):
    B, C, T = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype

    time_mask_ratio = float(max(0.0, min(1.0, time_mask_ratio)))
    num_preserved_blocks = max(1, int(num_preserved_blocks))

    if time_mask_ratio >= 1.0:
        mask = torch.zeros((B, 1, T), device=device, dtype=dtype).expand(-1, C, -1)
        return torch.zeros_like(input_tensor), mask
    if time_mask_ratio <= 0.0:
        mask = torch.ones((B, 1, T), device=device, dtype=dtype).expand(-1, C, -1)
        return input_tensor.clone(), mask

    target_visible = int(round((1 - time_mask_ratio) * T))
    target_visible = max(num_preserved_blocks * min_block_len, min(T, target_visible))

    base_len = target_visible // num_preserved_blocks
    remainder = target_visible - base_len * num_preserved_blocks
    lengths = torch.full((num_preserved_blocks,), base_len, device=device, dtype=torch.long)
    if remainder > 0:
        lengths[:remainder] += 1

    if base_len < min_block_len:
        lengths = torch.tensor([target_visible], device=device)
        num_blocks = 1
    else:
        num_blocks = num_preserved_blocks

    total_visible = int(lengths.sum().item())
    total_masked = T - total_visible
    if total_masked < 0:
        total_visible = T
        lengths = torch.tensor([T], device=device)
        num_blocks = 1
        total_masked = 0

    if total_masked == 0:
        gap_lengths = torch.zeros(num_blocks + 1, device=device, dtype=torch.long)
    else:
        if evenly_spaced:
            g_base = total_masked // (num_blocks + 1)
            g_rem = total_masked - g_base * (num_blocks + 1)
            gap_lengths = torch.full((num_blocks + 1,), g_base, device=device, dtype=torch.long)
            if g_rem > 0:
                gap_lengths[:g_rem] += 1
        else:
            raw = torch.rand(num_blocks + 1, device=device) + 1e-6
            gap_lengths = (raw / raw.sum() * total_masked).floor().long()
            diff = total_masked - gap_lengths.sum()
            if diff > 0:
                gap_lengths[:diff] += 1

    starts = []
    cursor = gap_lengths[0].item()
    for i in range(num_blocks):
        starts.append(cursor)
        cursor += lengths[i].item() + gap_lengths[i + 1].item()
    starts = torch.tensor(starts, device=device, dtype=torch.long)

    time_mask = torch.zeros(T, device=device, dtype=dtype)
    for s, L in zip(starts.tolist(), lengths.tolist()):
        time_mask[s: s + L] = 1

    mask = time_mask.view(1, 1, T).expand(B, C, -1).clone()
    masked_input = input_tensor * mask
    return masked_input, mask


def combine_time_channel_masks(
    x: torch.Tensor,
    time_mask_cfg: dict,
    channel_mask_ratio: float,
    channel_mask_mode: str,
    coords_for_topology: torch.Tensor | None = None,
):
    x_t, tmask = apply_mask_ssp(
        x,
        time_mask_ratio=float(time_mask_cfg.get("time_mask_ratio", 0.5)),
        num_preserved_blocks=int(time_mask_cfg.get("num_preserved_blocks", 8)),
        evenly_spaced=bool(time_mask_cfg.get("evenly_spaced", False)),
        min_block_len=int(time_mask_cfg.get("min_block_len", 4)),
    )

    cmask = random_channel_mask(
        x,
        channel_mask_ratio=float(channel_mask_ratio),
        mode=channel_mask_mode,
        min_keep=int(time_mask_cfg.get("min_keep_channels", 1)),
        coords=coords_for_topology,
        group_frac=float(time_mask_cfg.get("topo_group_frac", 0.25)),
    )

    maskTC = tmask * cmask
    x_masked = x * maskTC
    return x_masked, maskTC
