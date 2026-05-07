import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CORA(nn.Module):
    def __init__(
        self,
        coord_dim: int = 3,
        stat_dim: int = 5,
        model_dim: int = 128,
        num_heads: int = 4,
        coord_hidden_dim: int = 64,
        dropout: float = 0.0,
        eps: float = 1e-6,
        max_dist_bias_spread: float = 3.0,
        init_dist_bias_spread: float = 1.5,
        freeze_dist_scale: bool = False,
        attention_temp: float = 1.0,
        clamp_geo_bias: float = 2.0,
        freq_low: float = 0.1,
        freq_mid: float = 0.5,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by num_heads={num_heads}")

        self.coord_dim = coord_dim
        self.stat_dim = stat_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.eps = eps

        self.attention_temp = float(attention_temp)
        self.scale = (self.head_dim ** -0.5) / self.attention_temp
        self.clamp_geo_bias = float(clamp_geo_bias)
        self.freq_low = freq_low
        self.freq_mid = freq_mid

        self.max_dist_bias_spread = float(max_dist_bias_spread)
        p = max(min(init_dist_bias_spread / max_dist_bias_spread, 0.999), 0.001)
        raw_init = math.log(p / (1.0 - p))
        self.dist_spread_raw = nn.Parameter(torch.tensor(float(raw_init)))
        if freeze_dist_scale:
            self.dist_spread_raw.requires_grad_(False)
        self.freeze_dist_scale = freeze_dist_scale

        self.stat_proj = nn.Sequential(
            nn.Linear(stat_dim, model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim, model_dim),
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(coord_dim, coord_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(coord_hidden_dim, model_dim),
        )
        self.coord_bias_mlp = nn.Sequential(
            nn.Linear(coord_dim, coord_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(coord_hidden_dim, num_heads),
        )
        self._init_small_pos_proj()

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("_coord_center", torch.zeros(coord_dim), persistent=False)
        self.register_buffer("_coord_scale", torch.ones(1), persistent=False)
        self.register_buffer("_coord_stats_set", torch.tensor(False), persistent=False)

        self._has_sloreta = False
        self.register_buffer("sloreta_guide_W", torch.zeros(1), persistent=False)

        self._last_W = None
        self._last_W_heads = None

    def _init_small_pos_proj(self):
        for m in list(self.pos_proj.modules()) + list(self.coord_bias_mlp.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _coords_to_tensor(pos, device, coord_dim=3):
        if pos is None:
            return None
        if isinstance(pos, dict):
            names = list(pos.keys())
            P = torch.tensor([list(pos[n]) for n in names], dtype=torch.float32, device=device)
        elif torch.is_tensor(pos):
            P = pos.to(device=device, dtype=torch.float32)
        else:
            raise TypeError("coords must be dict, tensor, or None")

        if P.ndim != 2:
            raise ValueError(f"coords must be [C,D], got {P.shape}")
        if P.size(-1) < coord_dim:
            pad = torch.zeros(P.size(0), coord_dim - P.size(-1), device=device, dtype=P.dtype)
            P = torch.cat([P, pad], dim=-1)
        elif P.size(-1) > coord_dim:
            P = P[:, :coord_dim]
        return P

    def _set_coord_stats(self, all_pos):
        if self._coord_stats_set.item():
            return
        with torch.no_grad():
            center = all_pos.mean(dim=0)
            centered = all_pos - center
            if all_pos.size(0) >= 2:
                dist = torch.cdist(centered.unsqueeze(0), centered.unsqueeze(0))[0]
                n = dist.size(0)
                dist_flat = dist[~torch.eye(n, dtype=torch.bool, device=dist.device)]
                scale = dist_flat.mean().clamp(min=1e-3)
            else:
                scale = torch.tensor(1.0, device=all_pos.device)
            self._coord_center.copy_(center)
            self._coord_scale.copy_(scale.view(1))
            self._coord_stats_set.fill_(True)

    def _normalize_coords(self, pos):
        if not self._coord_stats_set.item():
            return pos
        return (pos - self._coord_center) / self._coord_scale

    def _channel_stats(self, x):
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)
        X_fft = torch.fft.rfft(x, dim=-1)
        power = X_fft.abs() ** 2
        total_power = power.sum(dim=-1) + self.eps
        F_bins = power.size(-1)
        low_end = max(1, int(F_bins * self.freq_low))
        mid_end = max(low_end + 1, int(F_bins * self.freq_mid))
        low_ratio = power[..., :low_end].sum(dim=-1) / total_power
        mid_ratio = power[..., low_end:mid_end].sum(dim=-1) / total_power
        high_ratio = power[..., mid_end:].sum(dim=-1) / total_power
        return torch.stack([mean, std, low_ratio, mid_ratio, high_ratio], dim=-1)

    def get_last_W(self, detach=True):
        if self._last_W is None:
            return None
        return self._last_W.detach() if detach else self._last_W

    def get_last_W_heads(self, detach=True):
        if self._last_W_heads is None:
            return None
        return self._last_W_heads.detach() if detach else self._last_W_heads

    def get_effective_dist_spread(self):
        return torch.sigmoid(self.dist_spread_raw) * self.max_dist_bias_spread

    def get_diagnostics(self):
        out = {
            "coord_center": self._coord_center.tolist(),
            "coord_scale": float(self._coord_scale.item()),
            "coord_stats_set": bool(self._coord_stats_set.item()),
            "dist_bias_spread": float(self.get_effective_dist_spread().item()),
            "has_sloreta_guide": bool(self._has_sloreta),
            "W_max_per_row_mean": None,
            "effective_k_mean": None,
            "per_head_effective_k": None,
        }
        if self._last_W is not None:
            W = self._last_W.detach()
            W_c = W.clamp(min=1e-12)
            ent = -(W_c * torch.log(W_c)).sum(dim=-1).mean()
            out["effective_k_mean"] = float(torch.exp(ent).item())
            out["W_max_per_row_mean"] = float(W.max(dim=-1).values.mean().item())
        if self._last_W_heads is not None:
            Wh = self._last_W_heads.detach()
            Wh_c = Wh.clamp(min=1e-12)
            ent_h = -(Wh_c * torch.log(Wh_c)).sum(dim=-1).mean(dim=(0, 2))
            out["per_head_effective_k"] = [float(torch.exp(e).item()) for e in ent_h]
        return out

    def compute_geo_guide_W(self, input_pos, target_pos, sigma_ratio=0.3):
        if not self._coord_stats_set.item():
            all_pos = torch.cat([input_pos, target_pos], dim=0)
            self._set_coord_stats(all_pos)

        input_n = self._normalize_coords(input_pos)
        target_n = self._normalize_coords(target_pos)
        coord_diff = target_n[:, None, :] - input_n[None, :, :]
        dist_sq = (coord_diff ** 2).sum(dim=-1)

        log_W = -dist_sq / (2 * (sigma_ratio ** 2))
        W_guide = torch.softmax(log_W, dim=-1)
        return W_guide

    def load_sloreta_guide(self, path_or_array):
        if isinstance(path_or_array, str):
            if not os.path.exists(path_or_array):
                raise FileNotFoundError(path_or_array)
            W = np.load(path_or_array)
        elif isinstance(path_or_array, np.ndarray):
            W = path_or_array
        elif torch.is_tensor(path_or_array):
            W = path_or_array.detach().cpu().numpy()
        else:
            raise TypeError(type(path_or_array))

        if W.ndim != 2:
            raise ValueError(f"sLORETA W must be 2D [C_roi, C_in], got {W.shape}")

        W = np.asarray(W, dtype=np.float32)
        W = np.clip(W, 0, None)
        row_sum = W.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0] = 1.0
        W = W / row_sum

        W_t = torch.tensor(W, dtype=torch.float32)
        self.sloreta_guide_W = W_t.to(self.sloreta_guide_W.device)
        self._has_sloreta = True
        print(f"[CORA] sLORETA guide loaded: {W_t.shape}")

    def has_sloreta_guide(self):
        return self._has_sloreta

    def forward(self, x, input_pos, target_pos, force_W=None):
        B, C_in, T = x.shape
        device = x.device

        input_pos = self._coords_to_tensor(input_pos, device, self.coord_dim)
        target_pos = self._coords_to_tensor(target_pos, device, self.coord_dim)
        if input_pos is None or target_pos is None:
            raise ValueError("CORA requires both input_pos and target_pos")

        C_out = target_pos.size(0)
        if input_pos.size(0) != C_in:
            raise ValueError(f"input_pos channels {input_pos.size(0)} != x channels {C_in}")

        if not self._coord_stats_set.item():
            all_pos = torch.cat([input_pos, target_pos], dim=0)
            self._set_coord_stats(all_pos)

        input_pos_n = self._normalize_coords(input_pos)
        target_pos_n = self._normalize_coords(target_pos)

        if force_W is not None:
            if force_W.ndim == 2:
                W = force_W.unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=x.dtype)
            elif force_W.ndim == 3:
                W = force_W.to(device=device, dtype=x.dtype)
            else:
                raise ValueError("force_W must be [C_out,C_in] or [B,C_out,C_in]")
            yS = torch.einsum("boc,bct->bot", W, x)
            self._last_W = W
            self._last_W_heads = None
            return yS, W

        stats = self._channel_stats(x)
        src_stat_feat = self.stat_proj(stats)
        src_pos_feat = self.pos_proj(input_pos_n).unsqueeze(0)
        src_tokens = src_stat_feat + src_pos_feat
        tgt_tokens = self.pos_proj(target_pos_n).unsqueeze(0).expand(B, -1, -1)

        Q = self.q_proj(tgt_tokens)
        K = self.k_proj(src_tokens)
        Q = Q.view(B, C_out, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, C_in, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_logits = torch.einsum("bhod,bhid->bhoi", Q, K) * self.scale

        coord_diff_n = target_pos_n[:, None, :] - input_pos_n[None, :, :]
        geo_bias = self.coord_bias_mlp(coord_diff_n)
        geo_bias = torch.clamp(geo_bias, -self.clamp_geo_bias, self.clamp_geo_bias)
        geo_bias = geo_bias.permute(2, 0, 1).unsqueeze(0)

        dist_n = torch.norm(coord_diff_n, dim=-1)
        dist_max = dist_n.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        dist_min = dist_n.min(dim=-1, keepdim=True).values
        dist_normalized = (dist_n - dist_min) / (dist_max - dist_min + 1e-6)
        spread = self.get_effective_dist_spread()
        dist_bias = -spread * dist_normalized
        dist_bias = dist_bias.unsqueeze(0).unsqueeze(0)

        attn_logits = attn_logits + geo_bias + dist_bias

        W_heads = torch.softmax(attn_logits, dim=-1)
        W_heads = self.dropout(W_heads)
        W = W_heads.mean(dim=1)

        yS = torch.einsum("boc,bct->bot", W, x)
        self._last_W = W
        self._last_W_heads = W_heads
        return yS, W


def cora_entropy_loss(W, target_entropy=2.5):
    W_c = W.clamp(min=1e-12)
    entropy = -(W_c * torch.log(W_c)).sum(dim=-1)
    return F.relu(target_entropy - entropy).mean()


def cora_locality_loss(W, input_pos, target_pos):
    device = W.device
    input_pos = input_pos.to(device=device, dtype=W.dtype)
    target_pos = target_pos.to(device=device, dtype=W.dtype)
    coord_diff = target_pos[:, None, :] - input_pos[None, :, :]
    dist = torch.norm(coord_diff, dim=-1).unsqueeze(0)
    return (W * dist).sum(dim=-1).mean()


def cora_head_diversity_loss(W_heads, weight=0.01):
    B, H, C_out, C_in = W_heads.shape
    if H < 2:
        return W_heads.new_zeros(())
    W_flat = W_heads.reshape(B, H, -1)
    W_norm = F.normalize(W_flat, dim=-1)
    sim = torch.einsum("bhd,bgd->bhg", W_norm, W_norm)
    mask = 1.0 - torch.eye(H, device=W_heads.device).unsqueeze(0)
    avg_sim = (sim * mask).sum() / (B * H * (H - 1) + 1e-8)
    return weight * avg_sim.clamp(min=0)


def cora_guide_loss(W, W_guide, mode="kl"):
    eps = 1e-12
    if W_guide.ndim == 2:
        W_guide_b = W_guide.unsqueeze(0).to(device=W.device, dtype=W.dtype)
    else:
        W_guide_b = W_guide.to(device=W.device, dtype=W.dtype)

    if mode == "kl":
        s = W.clamp(min=eps)
        t = W_guide_b.clamp(min=eps)
        loss = (s * (torch.log(s) - torch.log(t))).sum(dim=-1).mean()
    elif mode == "ce":
        s = W.clamp(min=eps)
        t = W_guide_b
        loss = -(t * torch.log(s)).sum(dim=-1).mean()
    elif mode == "l2":
        loss = ((W - W_guide_b) ** 2).sum(dim=-1).mean()
    else:
        raise ValueError(mode)
    return loss
