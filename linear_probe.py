from __future__ import annotations

import argparse
import copy
import gc
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _speller_loader():
    from src.utility.data_loader_speller import data_loader_speller
    return data_loader_speller


DATASET_REGISTRY: Dict[str, Callable[[Any], Dict]] = {
    "speller": lambda cfg: _speller_loader()(cfg),
}


def _peek_pretrain_out_channels(ckpt_path: str) -> Optional[int]:
    blob = torch.load(ckpt_path, map_location="cpu")
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    for k, v in sd.items():
        if k.endswith("final_conv.weight"):
            return int(v.shape[0])
    return None


def load_pretrained_module(ckpt_path: str, cfg, data: Dict, logger=None):
    from src.experiments.Reconstruction import ReconModule

    pretrain_out_c = _peek_pretrain_out_channels(ckpt_path)
    if pretrain_out_c is None:
        raise RuntimeError("Could not find final_conv.weight in ckpt")

    real_c = int(data["meta"].get("C", -1))
    if pretrain_out_c != real_c:
        print(f"[CKPT] pretrain_out={pretrain_out_c}, data C={real_c}; "
              f"building decoder at {pretrain_out_c} for ckpt match")

    patched_meta = dict(data["meta"])
    patched_meta["C"] = pretrain_out_c
    patched_data = dict(data)
    patched_data["meta"] = patched_meta

    module = ReconModule(cfg, patched_data, logger)

    blob = torch.load(ckpt_path, map_location="cpu")
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    missing, unexpected = module.load_state_dict(sd, strict=False)
    if missing:
        print(f"[CKPT] missing ({len(missing)}): {missing[:4]}{' ...' if len(missing) > 4 else ''}")
    if unexpected:
        print(f"[CKPT] unexpected ({len(unexpected)}): {unexpected[:4]}{' ...' if len(unexpected) > 4 else ''}")

    module.eval()
    return module


@torch.no_grad()
def extract_features(module, loader, device) -> np.ndarray:
    module.eval().to(device)
    feats = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.float().to(device, non_blocking=True)
        out = module(x, return_repr=True, return_weights=False)
        repr_t = out[1] if isinstance(out, (tuple, list)) else out
        if repr_t.dim() >= 3:
            repr_t = repr_t.flatten(1)
        feats.append(repr_t.detach().float().cpu())
    return torch.cat(feats, dim=0).numpy().astype(np.float32) if feats else np.zeros((0, 0), dtype=np.float32)


def fit_logreg_kfold(feats, labels, n_splits=5, seed=0,
                     C=1.0, max_iter=1000) -> Optional[Dict[str, Any]]:
    if feats.shape[0] < max(n_splits, 4):
        return None
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        return None
    n_splits_eff = int(min(n_splits, counts.min()))
    if n_splits_eff < 2:
        return None

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")),
    ])
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(feats, labels):
        pipe.fit(feats[tr], labels[tr])
        accs.append(float(pipe.score(feats[te], labels[te])))
    return {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "fold_acc": accs,
        "n_samples": int(len(labels)),
        "feat_dim": int(feats.shape[1]),
        "n_splits": n_splits_eff,
    }


def resolve_runs(args) -> List[Tuple[str, str]]:
    if args.dataset != "speller":
        return [("single", "")]
    versions = (args.speller_versions or "").strip()
    if not versions:
        return [("single", "")]
    if not args.speller_dir or not os.path.isdir(args.speller_dir):
        raise FileNotFoundError(f"--speller_dir not found: {args.speller_dir}")
    pat = args.speller_pattern or "Speller_{ver}.npz"
    out = []
    for ver in [v.strip() for v in versions.split(",") if v.strip()]:
        path = os.path.join(args.speller_dir, pat.format(ver=ver))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Speller npz not found for {ver!r}: {path}")
        out.append((ver, path))
    return out


def build_module_cfg(args) -> argparse.Namespace:
    return argparse.Namespace(
        model_name="CORABackbone",
        use_spatial_adapter=True,
        repr_mode=args.repr_mode,
        coord_dim=3,
        adapter_dim=args.adapter_dim,
        adapter_heads=args.adapter_heads,
        adapter_coord_hidden_dim=64,
        adapter_dropout=0.0,
        base_channels=args.base_channels,
        use_diff_latent=args.use_diff_latent,
        diff_num_heads=4, diff_mlp_ratio=2.0, diff_dropout=0.0,
        diff_smooth_kernel=5, diff_lambda_max=1.0,
        time_mask_ratio=0.0, channel_mask_ratio=0.0, channel_mask_mode="none",
        loss="L1Loss",
        cora_entropy_weight=0.0, cora_locality_weight=0.0,
        cora_head_div_weight=0.0, cora_target_entropy=2.5,
        guide_kind="none", guide_mode="kl",
        guide_geo_sigma=0.3, guide_geo_frac=0.5,
        sloreta_guide_path="",
        lr=1e-4, weight_decay=1e-2, epochs=1, onecycle_max_lr=5e-4,
    )


def evaluate_one(args, ver_tag: str, npz_override: str, device) -> Dict[str, Any]:
    cfg = copy.copy(args)
    if npz_override:
        cfg.speller_npz = npz_override
        cfg.speller_npz_files = ""
        cfg.speller_npz_dir = ""

    bundle = DATASET_REGISTRY[args.dataset](cfg)
    if bundle.get("labels") is None:
        raise RuntimeError(f"{args.dataset} returned no labels.")
    labels = np.asarray(bundle["labels"]).astype(np.int64).ravel()
    meta = bundle["meta"]
    print(f"[DATA] {meta.get('name')} | N={meta.get('N')} | C={meta.get('C')} "
          f"| T={meta.get('T_after')} | fs={meta.get('fs_hz')}")
    cls_str = ", ".join(f"{int(c)}:{int(n)}" for c, n in zip(*np.unique(labels, return_counts=True)))
    print(f"[DATA] classes: {{{cls_str}}}")

    module_cfg = build_module_cfg(args)
    if args.random_init:
        from src.experiments.Reconstruction import ReconModule
        module = ReconModule(module_cfg, bundle, logger=None).eval()
        print("[CKPT] random-init baseline")
    else:
        module = load_pretrained_module(args.ckpt_path, module_cfg, bundle, logger=None)

    feats = extract_features(module, bundle["loader"], device)
    if feats.shape[0] != labels.shape[0]:
        raise RuntimeError(f"feats/labels mismatch: {feats.shape} vs {labels.shape}")
    print(f"[FEAT] {feats.shape}")

    res = fit_logreg_kfold(feats, labels, n_splits=args.n_splits, seed=args.seed)
    if res is None:
        raise RuntimeError("Linear probe degenerate.")
    print(f"[PROBE] {args.dataset}/{ver_tag} | "
          f"mean={res['mean_acc']:.4f} +/- {res['std_acc']:.4f} | "
          f"folds={res['n_splits']} | N={res['n_samples']} | D={res['feat_dim']}")
    print(f"[PROBE] per-fold: {[f'{a:.4f}' for a in res['fold_acc']]}")

    del module, feats, bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "version": ver_tag,
        "dataset": args.dataset,
        "tag": args.tag,
        "ckpt_path": args.ckpt_path,
        "random_init": bool(args.random_init),
        "n_samples": res["n_samples"],
        "feat_dim": res["feat_dim"],
        "n_splits": res["n_splits"],
        "mean_acc": res["mean_acc"],
        "std_acc": res["std_acc"],
        "fold_acc": ";".join(f"{a:.6f}" for a in res["fold_acc"]),
    }


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY.keys()))
    p.add_argument("--ckpt_path", required=True)

    p.add_argument("--speller_npz", default="")
    p.add_argument("--speller_npz_files", default="")
    p.add_argument("--speller_npz_dir", default="")
    p.add_argument("--speller_dir", default="")
    p.add_argument("--speller_versions", default="")
    p.add_argument("--speller_pattern", default="Speller_{ver}.npz")

    p.add_argument("--split", default="all", choices=["train", "test", "all"])
    p.add_argument("--label_filter", default="all", choices=["all", "p300", "nonp300"])
    p.add_argument("--subjects", default="all")
    p.add_argument("--pad_multiple", type=int, default=16)

    p.add_argument("--input_coord_path", required=True)
    p.add_argument("--target_coord_path", required=True)

    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--repr_mode", default="bottleneck",
                   choices=["bottleneck", "multiscale", "tokens"])
    p.add_argument("--adapter_dim", type=int, default=128)
    p.add_argument("--adapter_heads", type=int, default=4)
    p.add_argument("--use_diff_latent", action="store_true")

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--random_init", action="store_true")

    p.add_argument("--out_csv", default="")
    p.add_argument("--tag", default="")
    return p


def main():
    args = build_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"[ENV] device={device} | dataset={args.dataset} | tag={args.tag or '(none)'}")

    runs = resolve_runs(args)
    print(f"[PLAN] {len(runs)} run(s): "
          + ", ".join(v if not p else f"{v}({os.path.basename(p)})" for v, p in runs))

    summaries = []
    for ver, path in runs:
        print(f"\n[RUN] === {ver} ===")
        s = evaluate_one(args, ver, path, device)
        summaries.append(s)
        if args.out_csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
            df = pd.DataFrame([s])
            if os.path.isfile(args.out_csv):
                df.to_csv(args.out_csv, mode="a", header=False, index=False)
            else:
                df.to_csv(args.out_csv, index=False)
            print(f"[OUT] appended to {args.out_csv}")

    print("\n[SUMMARY]")
    print(f"{'version':<10s} {'N':>6s} {'D':>5s} {'mean_acc':>10s} {'std':>8s}")
    for s in summaries:
        print(f"{s['version']:<10s} {s['n_samples']:>6d} {s['feat_dim']:>5d} "
              f"{s['mean_acc']:>10.4f} {s['std_acc']:>8.4f}")
    return summaries


if __name__ == "__main__":
    main()
