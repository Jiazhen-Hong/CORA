import argparse
import glob
import os

from pytorch_lightning import seed_everything

from src.experiments.Reconstruction import run_reconstruction
from src.utility.data_loader import data_loader


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--probe_dirs", type=str, required=True)

    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--crop_mode", type=str, default="around_pivot",
                        choices=["around_pivot", "tail", "none"])
    parser.add_argument("--win_pre_ms", type=int, default=0)
    parser.add_argument("--win_post_ms", type=int, default=1000)
    parser.add_argument("--tail_ms", type=int, default=400)
    parser.add_argument("--pad_multiple", type=int, default=16)
    parser.add_argument("--modality", type=str, default="all",
                        choices=["all", "vis", "aud"])
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "test", "all"])

    parser.add_argument("--input_coord_path", type=str,
                        default="./src/resources/montages/actiCAP64_dig_xyz.xlsx")
    parser.add_argument("--target_coord_path", type=str,
                        default="./src/resources/atlas/Schaefer300_7Networks_MNI_coords.xlsx")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--onecycle_max_lr", type=float, default=5e-4)

    parser.add_argument("--time_mask_ratio", type=float, default=0.0)
    parser.add_argument("--channel_mask_ratio", type=float, default=0.0)
    parser.add_argument("--channel_mask_mode", type=str, default="none",
                        choices=["none", "random", "topological"])
    parser.add_argument("--min_keep_channels", type=int, default=1)
    parser.add_argument("--ssp_blocks", type=int, default=8)
    parser.add_argument("--ssp_min_block", type=int, default=4)
    parser.add_argument("--ssp_even", action="store_true")
    parser.add_argument("--topo_group_frac", type=float, default=0.25)

    parser.add_argument("--loss", type=str, default="L1Loss",
                        choices=["L1Loss", "MSELoss"])

    parser.add_argument("--repr_mode", type=str, default="bottleneck",
                        choices=["bottleneck", "multiscale", "tokens"])
    parser.add_argument("--base_channels", type=int, default=32)

    parser.add_argument("--coord_dim", type=int, default=3)
    parser.add_argument("--adapter_dim", type=int, default=128)
    parser.add_argument("--adapter_heads", type=int, default=4)
    parser.add_argument("--adapter_coord_hidden_dim", type=int, default=64)
    parser.add_argument("--adapter_dropout", type=float, default=0.0)

    parser.add_argument("--use_diff_latent", action="store_true")
    parser.add_argument("--diff_num_heads", type=int, default=4)
    parser.add_argument("--diff_mlp_ratio", type=float, default=2.0)
    parser.add_argument("--diff_dropout", type=float, default=0.0)
    parser.add_argument("--diff_smooth_kernel", type=int, default=5)
    parser.add_argument("--diff_lambda_max", type=float, default=1.0)

    parser.add_argument("--cora_entropy_weight", type=float, default=0.05)
    parser.add_argument("--cora_locality_weight", type=float, default=0.001)
    parser.add_argument("--cora_head_div_weight", type=float, default=0.0)
    parser.add_argument("--cora_target_entropy", type=float, default=2.5)

    parser.add_argument("--guide_kind", type=str, default="none",
                        choices=["none", "geo", "sloreta", "both"])
    parser.add_argument("--sloreta_guide_path", type=str, default="")
    parser.add_argument("--guide_mode", type=str, default="kl",
                        choices=["kl", "ce", "l2"])
    parser.add_argument("--guide_geo_sigma", type=float, default=0.3)
    parser.add_argument("--guide_geo_frac", type=float, default=0.5)

    parser.add_argument("--guide_schedule", type=str, default="constant",
                        choices=["constant", "warmup_decay", "step"])
    parser.add_argument("--guide_weight_base", type=float, default=0.05)
    parser.add_argument("--guide_weight_warmup", type=float, default=0.3)
    parser.add_argument("--guide_warmup_epochs", type=int, default=15)
    parser.add_argument("--guide_decay_epochs", type=int, default=30)

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=3025)

    parser.add_argument("--out_dir", type=str, default="./Backbone/joint")
    parser.add_argument("--no_auto_suffix", action="store_true")
    parser.add_argument("--no_progress_bar", action="store_true")

    return parser


def compute_guide_weight_for_epoch(config, epoch):
    if config.guide_kind == "none":
        return 0.0

    schedule = config.guide_schedule
    w_base = config.guide_weight_base
    w_warm = config.guide_weight_warmup
    n_warm = config.guide_warmup_epochs
    n_decay = config.guide_decay_epochs

    if schedule == "constant":
        return w_base

    if schedule == "step":
        return w_warm if epoch < n_warm else w_base

    if schedule == "warmup_decay":
        if epoch < n_warm:
            return w_warm
        if epoch >= n_warm + n_decay:
            return w_base
        alpha = (epoch - n_warm) / max(1, n_decay)
        return w_warm + alpha * (w_base - w_warm)

    raise ValueError(f"Unknown guide_schedule: {schedule}")


def _mask_suffix(config):
    t = int(round(config.time_mask_ratio * 100))
    c = int(round(config.channel_mask_ratio * 100))
    base = f"mask{t}ch{c}"
    if config.guide_kind != "none":
        base += f"_guide-{config.guide_kind}"
    return base


def validate_config(config):
    config.use_spatial_adapter = True
    config.model_name = "CORABackbone"

    if not config.no_auto_suffix:
        suffix = _mask_suffix(config)
        base = config.out_dir.rstrip("/").rstrip(os.sep)
        if not base.endswith(f"_{suffix}"):
            config.out_dir = f"{base}_{suffix}"
    print(f"[OUT] out_dir = {config.out_dir}")

    dirs = [d.strip() for d in config.probe_dirs.split(",") if d.strip()]
    all_paths = []
    for d in dirs:
        found = sorted(glob.glob(os.path.join(d, "Probe*.npz")))
        if not found:
            print(f"[WARN] No Probe*.npz in: {d}")
        else:
            print(f"[INFO] {len(found)} probes from: {d}")
        all_paths.extend(found)
    if not all_paths:
        raise FileNotFoundError(f"No Probe*.npz found in any of: {dirs}")
    print(f"[INFO] Total: {len(all_paths)} probe files")

    config.probe_mode = "list"
    config.probe_files = ",".join(all_paths)
    config.probe_file = ""
    config.probe_dir = ""

    if config.guide_kind in ("sloreta", "both"):
        if not config.sloreta_guide_path:
            raise ValueError(
                f"--guide_kind={config.guide_kind} requires --sloreta_guide_path"
            )
        if not os.path.exists(config.sloreta_guide_path):
            raise FileNotFoundError(f"sLORETA guide not found: {config.sloreta_guide_path}")

    config.get_guide_weight = lambda epoch: compute_guide_weight_for_epoch(config, epoch)
    config.enable_progress_bar = not bool(getattr(config, "no_progress_bar", False))

    return config


def print_data_info(config, data):
    meta = data["meta"]
    print(
        f"[DATA] fs={meta.get('fs_hz')} | crop={meta.get('crop_mode')} | "
        f"T={meta.get('T_before')}->{meta.get('T_after')} | "
        f"modality={meta.get('modality')} | "
        f"N={meta.get('N')} | C={meta.get('C')} | "
        f"valid_ratio={meta.get('valid_ratio')}"
    )
    print(
        f"[COORD] input={len(data['input_coords'])} | "
        f"target={len(data['target_coords'])}"
    )
    pad_info = meta.get("pad_info")
    if pad_info:
        print(
            f"[PAD] left={pad_info.get('pad_left_ms', 0):.1f} ms | "
            f"right={pad_info.get('pad_right_ms', 0):.1f} ms"
        )
    print(
        f"[MODEL] repr_mode={config.repr_mode} | "
        f"base_channels={config.base_channels} | "
        f"use_diff_latent={config.use_diff_latent} | "
        f"adapter_dim={config.adapter_dim} | "
        f"adapter_heads={config.adapter_heads}"
    )
    print(
        f"[MASK] time={config.time_mask_ratio} | "
        f"channel={config.channel_mask_ratio} | "
        f"mode={config.channel_mask_mode}"
    )
    print(
        f"[CORA-REG] entropy={config.cora_entropy_weight} | "
        f"locality={config.cora_locality_weight} | "
        f"head_div={config.cora_head_div_weight} | "
        f"target_entropy={config.cora_target_entropy}"
    )
    print(
        f"[GUIDE] kind={config.guide_kind} | "
        f"mode={config.guide_mode} | "
        f"schedule={config.guide_schedule} | "
        f"base={config.guide_weight_base} | "
        f"warmup={config.guide_weight_warmup} "
        f"(for {config.guide_warmup_epochs} ep, decay {config.guide_decay_epochs} ep) | "
        f"sloreta_path={config.sloreta_guide_path or '(none)'}"
    )


def main():
    config = build_parser().parse_args()
    config = validate_config(config)

    seed_everything(config.seed, workers=True)

    data = data_loader(config)
    print_data_info(config, data)

    _, _, model_dir = run_reconstruction(config, data)
    print(f"[DONE] checkpoints: {model_dir}")
    print(f"[TB] tensorboard --logdir {config.out_dir}")


if __name__ == "__main__":
    main()
