import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_PROBE_RE = re.compile(r"Probe(\d+)\.npz$", re.IGNORECASE)
_LABEL_KEYS_TEMPLATE = [
    "y_{split}", "Y_{split}", "labels_{split}", "label_{split}",
    "y", "Y", "labels", "label",
]


def pad_to_multiple_keep_mask(x: np.ndarray, mask: np.ndarray, multiple: int) -> Tuple[np.ndarray, np.ndarray]:
    if multiple <= 1 or x.shape[-1] % multiple == 0:
        return x, mask
    pad = multiple - (x.shape[-1] % multiple)
    return (
        np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="constant"),
        np.pad(mask, ((0, 0), (0, pad)), mode="constant"),
    )


def infer_pivot_ms_from_name(path: str) -> Optional[int]:
    match = _PROBE_RE.search(os.path.basename(path))
    return int(match.group(1)) if match else None


def crop_around_pivot(x: np.ndarray, fs_hz: float, pivot_ms: int, win_pre_ms: int, win_post_ms: int):
    if x.ndim != 3:
        raise ValueError(f"Expect [N,C,T], got {x.shape}")

    n_samples, _, total_t = x.shape
    pivot_samp = int(round(fs_hz * pivot_ms / 1000.0))
    pre_samp = int(round(fs_hz * win_pre_ms / 1000.0))
    post_samp = int(round(fs_hz * win_post_ms / 1000.0))
    window_t = pre_samp + post_samp

    start = pivot_samp - pre_samp
    end = pivot_samp + post_samp
    valid_start = max(0, start)
    valid_end = min(total_t, end)

    x_crop = x[..., valid_start:valid_end]
    pad_left = max(0, -start)
    pad_right = max(0, end - total_t)
    if pad_left or pad_right or x_crop.shape[-1] != window_t:
        extra_right = max(0, window_t - x_crop.shape[-1] - pad_left - pad_right)
        x_crop = np.pad(x_crop, ((0, 0), (0, 0), (pad_left, pad_right + extra_right)), mode="constant")
        x_crop = x_crop[..., :window_t]

    mask = np.zeros((n_samples, window_t), dtype=np.float32)
    mask[:, pad_left:window_t - pad_right] = 1.0
    pad_meta = {
        "pad_left_samp": int(pad_left),
        "pad_right_samp": int(pad_right),
        "pad_left_ms": float(1000.0 * pad_left / fs_hz),
        "pad_right_ms": float(1000.0 * pad_right / fs_hz),
        "Twin": window_t,
    }
    return x_crop.astype(np.float32), mask, pad_meta


def crop_tail_ms(x: np.ndarray, fs_hz: float, tail_ms: int):
    tail_samples = max(1, int(round(fs_hz * tail_ms / 1000.0)))
    if x.shape[-1] < tail_samples:
        raise ValueError(f"T={x.shape[-1]} < tail_samples={tail_samples}")
    x_crop = x[..., -tail_samples:].astype(np.float32)
    mask = np.ones((x.shape[0], tail_samples), dtype=np.float32)
    pad_meta = {"pad_left_samp": 0, "pad_right_samp": 0, "pad_left_ms": 0.0, "pad_right_ms": 0.0, "Twin": tail_samples}
    return x_crop, mask, pad_meta


def load_labels(pack: np.lib.npyio.NpzFile, split: str) -> Optional[np.ndarray]:
    for template in _LABEL_KEYS_TEMPLATE:
        key = template.format(split=split)
        if key in pack.files:
            labels = np.asarray(pack[key]).reshape(-1)
            try:
                labels = labels.astype(np.int64)
            except Exception:
                pass
            return labels
    return None


def load_probe_npz(path: str, split: str) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    pack = np.load(path, allow_pickle=True)

    if split == "train":
        x = pack["X_train"].astype(np.float32)
        labels = load_labels(pack, "train")
    elif split == "test":
        x = pack["X_test"].astype(np.float32)
        labels = load_labels(pack, "test")
    elif split == "all":
        if "X_train" not in pack.files or "X_test" not in pack.files:
            raise ValueError(f"{path} missing X_train/X_test for split='all'")
        x_train = pack["X_train"].astype(np.float32)
        x_test = pack["X_test"].astype(np.float32)
        x = np.concatenate([x_train, x_test], axis=0)

        y_train = load_labels(pack, "train")
        y_test = load_labels(pack, "test")
        if y_train is not None and y_test is not None:
            labels = np.concatenate([y_train, y_test], axis=0)
        else:
            labels = None
    else:
        raise ValueError(f"Unknown split: {split}")

    pivot_ms = None
    for k in ("stimuli_ms", "pivot_ms", "stim_ms"):
        if k in pack.files:
            try:
                pivot_ms = int(pack[k])
            except Exception:
                pass
            break

    fs_hz = None
    if "fs" in pack.files:
        try:
            fs_hz = float(pack["fs"])
        except Exception:
            pass

    meta = {
        "name": pack["name"].item() if "name" in pack.files else os.path.basename(path),
        "target_len": int(pack["target_len"]) if "target_len" in pack.files else x.shape[-1],
        "N": x.shape[0],
        "C": x.shape[1],
        "T_before": x.shape[-1],
        "T_after": x.shape[-1],
        "path": os.path.abspath(path),
        "pivot_ms": pivot_ms if pivot_ms is not None else infer_pivot_ms_from_name(path),
        "has_labels": labels is not None,
        "fs_hz_in_npz": fs_hz,
        "split": split,
    }
    return x, labels, meta


def df_to_coord_dict(df: pd.DataFrame) -> Dict[str, tuple]:
    cols = {c.lower(): c for c in df.columns}
    if not all(k in cols for k in ("x", "y", "z")):
        raise ValueError("Excel must have columns x,y,z")

    if "name" in cols:
        name_col = cols["name"]
        return {
            str(row[name_col]): (
                float(row[cols["x"]]),
                float(row[cols["y"]]),
                float(row[cols["z"]]),
            )
            for _, row in df.iterrows()
        }
    return {
        str(idx): (
            float(df.at[idx, cols["x"]]),
            float(df.at[idx, cols["y"]]),
            float(df.at[idx, cols["z"]]),
        )
        for idx in df.index
    }


class ArrayReconDataset(Dataset):
    def __init__(self, x_np: np.ndarray):
        if x_np.ndim != 3:
            raise ValueError(f"Expect [N,C,T], got {x_np.shape}")
        self.x = torch.from_numpy(x_np.astype(np.float32))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        return x, x


def apply_modality_filter(x: np.ndarray, mask: np.ndarray, y: Optional[np.ndarray], modality: str):
    stats = {"N_before": x.shape[0], "N_after": x.shape[0], "n_vis": None, "n_aud": None}
    if y is None or modality == "all":
        if y is not None:
            stats["n_vis"] = int((y == 0).sum())
            stats["n_aud"] = int((y == 1).sum())
        return x, mask, y, stats

    if modality not in {"vis", "aud"}:
        raise ValueError(f"Unknown modality {modality}")

    target_label = 0 if modality == "vis" else 1
    idx = np.where(y == target_label)[0]
    if idx.size == 0:
        raise ValueError(f"No samples for modality='{modality}'")

    x = x[idx]
    mask = mask[idx]
    y = y[idx]
    stats.update({
        "N_after": x.shape[0],
        "n_vis": int((y == 0).sum()),
        "n_aud": int((y == 1).sum()),
    })
    return x, mask, y, stats


def gather_probe_paths(config) -> List[str]:
    mode = getattr(config, "probe_mode", "single")
    if mode == "single":
        probe_file = getattr(config, "probe_file", None)
        if not probe_file:
            raise ValueError("--probe_mode=single requires --probe_file")
        return [probe_file]

    if mode == "all":
        folder = getattr(config, "probe_dir", None) or os.path.dirname(getattr(config, "probe_file", ""))
        if not folder or not os.path.isdir(folder):
            raise FileNotFoundError(f"Probe directory not found: {folder}")
        paths = sorted(glob.glob(os.path.join(folder, "Probe*.npz")))
        if not paths:
            raise FileNotFoundError(f"No Probe*.npz found in {folder}")
        return paths

    if mode == "list":
        raw = getattr(config, "probe_files", "")
        if not raw:
            raise ValueError("--probe_mode=list requires --probe_files")
        items = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            matched = glob.glob(token)
            if matched:
                items.extend(matched)
            elif os.path.isfile(token):
                items.append(token)
            else:
                raise FileNotFoundError(f"Not found: {token}")
        paths = sorted(set(items))
        if not paths:
            raise FileNotFoundError("Empty --probe_files after expansion")
        return paths

    raise ValueError(f"Unknown probe_mode: {mode}")


def crop_by_mode(x, fs_hz, crop_mode, pivot_ms, win_pre_ms, win_post_ms, tail_ms):
    if crop_mode == "around_pivot":
        if pivot_ms is None:
            raise ValueError("Cannot infer pivot (stimuli ms)")
        return crop_around_pivot(x, fs_hz, pivot_ms, win_pre_ms, win_post_ms)
    if crop_mode == "tail":
        return crop_tail_ms(x, fs_hz, tail_ms)
    if crop_mode == "none":
        mask = np.ones((x.shape[0], x.shape[-1]), dtype=np.float32)
        pad_meta = {"pad_left_samp": 0, "pad_right_samp": 0, "pad_left_ms": 0.0, "pad_right_ms": 0.0, "Twin": x.shape[-1]}
        return x.astype(np.float32), mask, pad_meta
    raise ValueError(f"Unknown crop_mode: {crop_mode}")


def load_and_harmonize_probes(paths, split, pad_multiple, fs_hz, crop_mode, tail_ms, win_pre_ms, win_post_ms, modality):
    arrays, masks, labels_list, files_meta = [], [], [], []
    c_ref = None
    n_vis_total, n_aud_total = 0, 0

    for path in paths:
        x, y, meta = load_probe_npz(path, split)
        fs_in = meta.get("fs_hz_in_npz")
        if fs_in is not None and abs(float(fs_in) - float(fs_hz)) > 1e-6:
            raise ValueError(f"fs mismatch: {path} has fs={fs_in}, but global fs={fs_hz}")
        if c_ref is None:
            c_ref = x.shape[1]
        elif x.shape[1] != c_ref:
            raise ValueError(f"Channel mismatch: {path} has C={x.shape[1]}, expected {c_ref}")

        pivot_ms = meta.get("pivot_ms") or infer_pivot_ms_from_name(path)
        x, mask, pad_meta = crop_by_mode(x, fs_hz, crop_mode, pivot_ms, win_pre_ms, win_post_ms, tail_ms)
        x, mask, y, stats = apply_modality_filter(x, mask, y, modality)

        if stats["n_vis"] is not None:
            n_vis_total += stats["n_vis"]
        if stats["n_aud"] is not None:
            n_aud_total += stats["n_aud"]

        arrays.append(x)
        masks.append(mask)
        labels_list.append(y)
        files_meta.append({
            "path": meta["path"],
            "N": meta["N"],
            "C": meta["C"],
            "T_before": meta["T_before"],
            "T_after": meta["T_after"],
            "T_after_crop": x.shape[-1],
            "pivot_ms": pivot_ms,
            "N_after_filter": x.shape[0],
            "pad_info": pad_meta,
        })

    max_t = max(arr.shape[-1] for arr in arrays)
    padded_arrays, padded_masks = [], []
    for x, mask in zip(arrays, masks):
        if x.shape[-1] < max_t:
            pad = max_t - x.shape[-1]
            x = np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="constant")
            mask = np.pad(mask, ((0, 0), (0, pad)), mode="constant")
        x, mask = pad_to_multiple_keep_mask(x, mask, pad_multiple)
        padded_arrays.append(x)
        padded_masks.append(mask)

    x_all = np.concatenate(padded_arrays, axis=0)
    mask_all = np.concatenate(padded_masks, axis=0)
    y_all = (
        np.concatenate([lab for lab in labels_list if lab is not None], axis=0)
        if any(lab is not None for lab in labels_list)
        else None
    )

    meta = {
        "name": f"MultiProbe[{len(paths)}]_{crop_mode}",
        "files": files_meta,
        "N": x_all.shape[0],
        "C": c_ref,
        "T_after": x_all.shape[-1],
        "target_len": x_all.shape[-1],
        "crop_mode": crop_mode,
        "fs_hz": fs_hz,
        "win_pre_ms": win_pre_ms,
        "win_post_ms": win_post_ms,
        "tail_ms": tail_ms,
        "modality": modality,
        "split": split,
        "n_vis": n_vis_total if (n_vis_total + n_aud_total) > 0 else None,
        "n_aud": n_aud_total if (n_vis_total + n_aud_total) > 0 else None,
        "valid_ratio": float(mask_all.sum() / mask_all.size),
        "has_labels": y_all is not None,
    }
    return x_all, mask_all, y_all, meta


def data_loader(config) -> Dict:
    paths = gather_probe_paths(config)
    fs_hz = float(getattr(config, "fs", 250))
    crop_mode = getattr(config, "crop_mode", "around_pivot")
    tail_ms = getattr(config, "tail_ms", 1000)
    win_pre_ms = getattr(config, "win_pre_ms", 200)
    win_post_ms = getattr(config, "win_post_ms", 1000)
    modality = getattr(config, "modality", "all")
    pad_multiple = getattr(config, "pad_multiple", 16)

    if len(paths) == 1:
        x, y, meta = load_probe_npz(paths[0], config.split)
        if meta.get("fs_hz_in_npz") is not None:
            fs_hz = float(meta["fs_hz_in_npz"])
        pivot_ms = meta.get("pivot_ms") or infer_pivot_ms_from_name(paths[0])
        x, mask, pad_meta = crop_by_mode(x, fs_hz, crop_mode, pivot_ms, win_pre_ms, win_post_ms, tail_ms)
        x, mask, y, stats = apply_modality_filter(x, mask, y, modality)
        x, mask = pad_to_multiple_keep_mask(x, mask, pad_multiple)

        meta.update({
            "split": config.split,
            "crop_mode": crop_mode,
            "fs_hz": fs_hz,
            "win_pre_ms": win_pre_ms,
            "win_post_ms": win_post_ms,
            "tail_ms": tail_ms,
            "T_after": x.shape[-1],
            "target_len": x.shape[-1],
            "modality": modality,
            "n_vis": stats["n_vis"],
            "n_aud": stats["n_aud"],
            "N_before": stats["N_before"],
            "N_after_filter": x.shape[0],
            "N_raw": meta.get("N"),
            "N": x.shape[0],
            "pad_info": pad_meta,
            "valid_ratio": float(mask.sum() / mask.size),
            "has_labels": y is not None,
        })
        x_all, mask_all, y_all = x, mask, y

    else:
        first_meta = load_probe_npz(paths[0], config.split)[2]
        if first_meta.get("fs_hz_in_npz") is not None:
            fs_hz = float(first_meta["fs_hz_in_npz"])
        x_all, mask_all, y_all, meta = load_and_harmonize_probes(
            paths, config.split, pad_multiple, fs_hz, crop_mode, tail_ms, win_pre_ms, win_post_ms, modality
        )

    dataset = ArrayReconDataset(x_all)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    input_coords = df_to_coord_dict(pd.read_excel(config.input_coord_path, index_col=0))
    target_coords = df_to_coord_dict(pd.read_excel(config.target_coord_path, index_col=0))

    return {
        "loader": dataloader,
        "meta": meta,
        "input_coords": input_coords,
        "target_coords": target_coords,
        "valid_mask": mask_all,
        "labels": y_all,
    }
