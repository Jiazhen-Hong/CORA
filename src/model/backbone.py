import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.cora_adapter import (
    CORA,
    cora_entropy_loss,
    cora_locality_loss,
    cora_head_diversity_loss,
    cora_guide_loss,
)


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )


def light_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                            padding=padding, groups=channels, bias=False)
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.pw(self.dw(x)))))


class DifferentialLatentRefiner(nn.Module):
    def __init__(self, channels, num_heads=4, mlp_ratio=2.0, dropout=0.0,
                 smooth_kernel=5, lambda_max=1.0):
        super().__init__()
        if smooth_kernel % 2 == 0:
            raise ValueError("smooth_kernel must be odd")
        self.channels = channels
        self.lambda_max = lambda_max

        self.main_branch = nn.Sequential(
            DepthwiseSeparableConv1d(channels, kernel_size=3, dropout=dropout),
            DepthwiseSeparableConv1d(channels, kernel_size=3, dropout=dropout),
        )
        self.ref_smoother = nn.AvgPool1d(smooth_kernel, stride=1, padding=smooth_kernel // 2)
        self.ref_branch = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.main_gate = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.ref_gate = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.out_norm = nn.BatchNorm1d(channels)
        self.raw_lambda = nn.Parameter(torch.tensor(0.0))
        self._last_parts = {}

    def get_lambda(self):
        return torch.sigmoid(self.raw_lambda) * self.lambda_max

    def get_last_parts(self, detach=True):
        if not self._last_parts:
            return None
        if not detach:
            return self._last_parts
        return {k: (v.detach() if torch.is_tensor(v) else v)
                for k, v in self._last_parts.items()}

    def forward(self, H):
        H_main_delta = self.main_branch(H)
        H_main = H + torch.tanh(self.main_gate(H_main_delta)) * H_main_delta
        H_smooth = self.ref_smoother(H)
        H_ref_delta = self.ref_branch(H_smooth)
        H_ref = H_smooth + torch.tanh(self.ref_gate(H_ref_delta)) * H_ref_delta

        lam = self.get_lambda()
        H_diff = self.out_norm(H_main - lam * H_ref)
        self._last_parts = {"main": H_main, "ref": H_ref, "diff": H_diff, "lambda": lam}
        return H_diff


class CORABackbone(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        repr_mode="bottleneck",
        use_spatial_adapter=False,
        input_coords=None,
        target_coords=None,
        coord_dim=3,
        adapter_dim=128,
        adapter_heads=4,
        adapter_coord_hidden_dim=64,
        adapter_dropout=0.0,
        base_channels=32,
        use_diff_latent=True,
        diff_num_heads=4,
        diff_mlp_ratio=2.0,
        diff_dropout=0.0,
        diff_smooth_kernel=5,
        diff_lambda_max=1.0,
        cora_max_dist_bias_spread=3.0,
        cora_init_dist_bias_spread=1.5,
        cora_freeze_dist_scale=False,
        cora_attention_temp=1.0,
        cora_clamp_geo_bias=2.0,
        cora_freq_low=0.1,
        cora_freq_mid=0.5,
    ):
        super().__init__()
        if repr_mode not in {"bottleneck", "multiscale", "tokens"}:
            raise ValueError(f"Unsupported repr_mode={repr_mode}")

        self.repr_mode = repr_mode
        self.use_spatial_adapter = use_spatial_adapter
        self.backbone_in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.use_diff_latent = use_diff_latent

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        cb = base_channels * 8

        self.input_all_names = list(input_coords.keys()) if isinstance(input_coords, dict) else []
        self.target_all_names = list(target_coords.keys()) if isinstance(target_coords, dict) else []

        in_pos_buf = self._coords_to_tensor_or_empty(input_coords, coord_dim)
        out_pos_buf = self._coords_to_tensor_or_empty(target_coords, coord_dim)
        self.register_buffer("in_pos_buf", in_pos_buf, persistent=False)
        self.register_buffer("out_pos_buf", out_pos_buf, persistent=False)

        if self.use_spatial_adapter:
            if self.out_pos_buf.numel() == 0:
                raise ValueError("use_spatial_adapter=True requires target_coords")
            if self.out_pos_buf.size(0) != in_channels:
                raise ValueError(
                    f"target_coords count ({self.out_pos_buf.size(0)}) != "
                    f"in_channels ({in_channels})"
                )
            self.cora = CORA(
                coord_dim=coord_dim,
                model_dim=adapter_dim,
                num_heads=adapter_heads,
                coord_hidden_dim=adapter_coord_hidden_dim,
                dropout=adapter_dropout,
                max_dist_bias_spread=cora_max_dist_bias_spread,
                init_dist_bias_spread=cora_init_dist_bias_spread,
                freeze_dist_scale=cora_freeze_dist_scale,
                attention_temp=cora_attention_temp,
                clamp_geo_bias=cora_clamp_geo_bias,
                freq_low=cora_freq_low,
                freq_mid=cora_freq_mid,
            )
        else:
            self.cora = None

        self.encoder1 = conv_block(in_channels, c1)
        self.pool1 = nn.MaxPool1d(2)
        self.encoder2 = conv_block(c1, c2)
        self.pool2 = nn.MaxPool1d(2)
        self.encoder3 = conv_block(c2, c3)
        self.pool3 = nn.MaxPool1d(2)
        self.bottleneck = conv_block(c3, cb)

        if self.use_diff_latent:
            self.diff_latent = DifferentialLatentRefiner(
                channels=cb,
                num_heads=diff_num_heads,
                mlp_ratio=diff_mlp_ratio,
                dropout=diff_dropout,
                smooth_kernel=diff_smooth_kernel,
                lambda_max=diff_lambda_max,
            )
        else:
            self.diff_latent = None

        self.upconv3 = nn.ConvTranspose1d(cb, c3, kernel_size=2, stride=2)
        self.decoder3 = light_conv_block(c3 + c3, c3)
        self.upconv2 = nn.ConvTranspose1d(c3, c2, kernel_size=2, stride=2)
        self.decoder2 = light_conv_block(c2 + c2, c2)
        self.upconv1 = nn.ConvTranspose1d(c2, c1, kernel_size=2, stride=2)
        self.decoder1 = light_conv_block(c1 + c1, c1)
        self.final_conv = nn.Conv1d(c1, out_channels, kernel_size=1)

    @staticmethod
    def gap(x):
        return F.adaptive_avg_pool1d(x, 1).squeeze(-1)

    @staticmethod
    def gmp(x):
        return F.adaptive_max_pool1d(x, 1).squeeze(-1)

    @staticmethod
    def _coords_to_tensor_or_empty(pos, coord_dim=3):
        if pos is None:
            return torch.empty(0, coord_dim, dtype=torch.float32)
        if isinstance(pos, dict):
            P = torch.tensor([list(pos[k]) for k in pos.keys()], dtype=torch.float32)
        elif torch.is_tensor(pos):
            P = pos.detach().cpu().float()
        else:
            raise TypeError("coords must be dict, tensor, or None")
        if P.ndim != 2:
            raise ValueError(f"coords must be [C,D], got {P.shape}")
        if P.size(-1) < coord_dim:
            pad = torch.zeros(P.size(0), coord_dim - P.size(-1), dtype=P.dtype)
            P = torch.cat([P, pad], dim=-1)
        elif P.size(-1) > coord_dim:
            P = P[:, :coord_dim]
        return P

    @staticmethod
    def _match_length(x, ref):
        tx = x.size(-1)
        tr = ref.size(-1)
        if tx == tr:
            return x
        if tx < tr:
            pad_total = tr - tx
            pl, pr = pad_total // 2, pad_total - pad_total // 2
            return F.pad(x, (pl, pr))
        cl = (tx - tr) // 2
        cr = (tx - tr) - cl
        return x[..., cl: tx - cr]

    def cora_route(self, x, input_coords=None, target_coords=None, force_W=None):
        if not self.use_spatial_adapter:
            raise RuntimeError("cora_route() called but use_spatial_adapter=False")
        if input_coords is None:
            if self.in_pos_buf.numel() == 0:
                raise ValueError("No default input_coords found")
            input_coords = self.in_pos_buf
        if target_coords is None:
            if self.out_pos_buf.numel() == 0:
                raise ValueError("No default target_coords found")
            target_coords = self.out_pos_buf
        return self.cora(x, input_coords, target_coords, force_W=force_W)

    def get_last_W(self, detach=True):
        return None if self.cora is None else self.cora.get_last_W(detach=detach)

    def get_last_W_heads(self, detach=True):
        return None if self.cora is None else self.cora.get_last_W_heads(detach=detach)

    def get_cora_diagnostics(self):
        return None if self.cora is None else self.cora.get_diagnostics()

    def get_last_diff_parts(self, detach=True):
        if not self.use_diff_latent or self.diff_latent is None:
            return None
        return self.diff_latent.get_last_parts(detach=detach)

    def load_sloreta_guide(self, path_or_array):
        if self.cora is None:
            raise RuntimeError("No CORA adapter; cannot load guide.")
        self.cora.load_sloreta_guide(path_or_array)

    def compute_cora_losses(
        self,
        entropy_weight: float = 0.05,
        locality_weight: float = 0.001,
        head_diversity_weight: float = 0.0,
        target_entropy: float = 2.5,
        guide_kind: str = "none",
        guide_weight: float = 0.0,
        guide_geo_sigma: float = 0.3,
        guide_mode: str = "kl",
        guide_geo_frac: float = 0.5,
        input_coords=None,
        target_coords=None,
    ):
        device = next(self.parameters()).device
        if self.cora is None:
            return torch.tensor(0.0, device=device)

        W = self.cora.get_last_W(detach=False)
        if W is None:
            return torch.tensor(0.0, device=device)

        loss = W.new_zeros(())

        if entropy_weight > 0:
            loss = loss + entropy_weight * cora_entropy_loss(W, target_entropy=target_entropy)

        if locality_weight > 0:
            ic = self.in_pos_buf if input_coords is None else input_coords
            tc = self.out_pos_buf if target_coords is None else target_coords
            if ic.numel() > 0 and tc.numel() > 0:
                loss = loss + locality_weight * cora_locality_loss(W, ic, tc)

        if head_diversity_weight > 0:
            Wh = self.cora.get_last_W_heads(detach=False)
            if Wh is not None:
                loss = loss + cora_head_diversity_loss(Wh, weight=head_diversity_weight)

        if guide_weight > 0 and guide_kind != "none":
            ic = self.in_pos_buf if input_coords is None else input_coords
            tc = self.out_pos_buf if target_coords is None else target_coords
            ic_dev = ic.to(W.device)
            tc_dev = tc.to(W.device)

            if guide_kind == "geo":
                W_g = self.cora.compute_geo_guide_W(ic_dev, tc_dev, sigma_ratio=guide_geo_sigma)
                loss = loss + guide_weight * cora_guide_loss(W, W_g, mode=guide_mode)

            elif guide_kind == "sloreta":
                if not self.cora.has_sloreta_guide():
                    raise RuntimeError(
                        "guide_kind='sloreta' but no sLORETA guide loaded. "
                        "Call model.load_sloreta_guide(path) first."
                    )
                W_g = self.cora.sloreta_guide_W.to(W.device)
                loss = loss + guide_weight * cora_guide_loss(W, W_g, mode=guide_mode)

            elif guide_kind == "both":
                W_geo = self.cora.compute_geo_guide_W(ic_dev, tc_dev, sigma_ratio=guide_geo_sigma)
                w_geo = guide_weight * guide_geo_frac
                loss = loss + w_geo * cora_guide_loss(W, W_geo, mode=guide_mode)

                if self.cora.has_sloreta_guide():
                    W_slo = self.cora.sloreta_guide_W.to(W.device)
                    w_slo = guide_weight * (1.0 - guide_geo_frac)
                    loss = loss + w_slo * cora_guide_loss(W, W_slo, mode=guide_mode)
                else:
                    print("[WARN] guide_kind='both' but no sLORETA loaded; using geo only.")

            else:
                raise ValueError(f"Unknown guide_kind={guide_kind}")

        return loss

    def forward(self, x, return_repr=False, input_coords=None,
                target_coords=None, return_weights=False, force_W=None):
        W = None
        if self.use_spatial_adapter:
            x, W = self.cora_route(
                x, input_coords=input_coords, target_coords=target_coords, force_W=force_W,
            )
        else:
            if x.shape[1] != self.backbone_in_channels:
                raise ValueError(
                    f"backbone expects {self.backbone_in_channels} channels, got {x.shape[1]}"
                )

        x1 = self.encoder1(x)
        x1p = self.pool1(x1)
        x2 = self.encoder2(x1p)
        x2p = self.pool2(x2)
        x3 = self.encoder3(x2p)
        x3p = self.pool3(x3)
        x4 = self.bottleneck(x3p)
        x4_refined = self.diff_latent(x4) if self.use_diff_latent else x4

        representation = None
        if return_repr:
            if self.repr_mode == "bottleneck":
                representation = self.gap(x4_refined)
            elif self.repr_mode == "multiscale":
                representation = torch.cat([self.gap(x1p), self.gap(x2p), self.gap(x3p)], dim=1)
            else:
                representation = x4_refined

        x = self.upconv3(x4_refined)
        x = self._match_length(x, x3)
        x = self.decoder3(torch.cat((x3, x), dim=1))
        x = self.upconv2(x)
        x = self._match_length(x, x2)
        x = self.decoder2(torch.cat((x2, x), dim=1))
        x = self.upconv1(x)
        x = self._match_length(x, x1)
        x = self.decoder1(torch.cat((x1, x), dim=1))

        out = self.final_conv(x)
        out = self._match_length(out, x.new_zeros(x.size(0), self.out_channels, x1.size(-1)))

        if return_repr and return_weights:
            return out, representation, W
        if return_repr:
            return out, representation
        if return_weights:
            return out, W
        return out
