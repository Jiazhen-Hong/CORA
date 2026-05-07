from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def _pad_time_to_multiple(x: np.ndarray, multiple: int) -> np.ndarray:
    if multiple <= 1 or x.shape[-1] % multiple == 0:
        return x
    pad = multiple - (x.shape[-1] % multiple)
    return np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="constant")


def _df_to_coord_dict(df: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    cols = {c.lower(): c for c in df.columns}
    if not all(k in cols for k in ("x", "y", "z")):
        raise ValueError("Coord excel must have columns x,y,z")

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
        self.x = torch.from_numpy(np.ascontiguousarray(x_np.astype(np.float32)))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        return x, x


def _select_split(pack: np.lib.npyio.NpzFile, split: str
                  ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    files_lower = {k.lower(): k for k in pack.files}

    def _get(key_l: str):
        return pack[files_lower[key_l]] if key_l in files_lower else None

    if split == "train":
        x = _get("x_train")
        y = _get("y_train")
        s = _get("subjects_train")
    elif split == "test":
        x = _get("x_test")
        y = _get("y_test")
        s = _get("subjects_test")
    elif split == "all":
        x_tr, x_te = _get("x_train"), _get("x_test")
        if x_tr is None and x_te is None:
            raise ValueError("npz has neither X_train nor X_test")
        x = (
            np.concatenate([x_tr, x_te], axis=0)
            if (x_tr is not None and x_te is not None) else
            (x_tr if x_tr is not None else x_te)
        )
        y_tr, y_te = _get("y_train"), _get("y_test")
        if y_tr is not None and y_te is not None:
            y = np.concatenate([y_tr, y_te], axis=0)
        elif y_tr is not None:
            y = y_tr
        elif y_te is not None:
            y = y_te
        else:
            y = None

        s_tr, s_te = _get("subjects_train"), _get("subjects_test")
        if s_tr is not None and s_te is not None:
            s = np.concatenate([s_tr, s_te], axis=0)
        elif s_tr is not None:
            s = s_tr
        elif s_te is not None:
            s = s_te
        else:
            s = None
    else:
        raise ValueError(f"Unknown split: {split!r}")

    if x is None:
        raise ValueError(f"No X data found for split={split!r}")

    x = np.asarray(x).astype(np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected X with shape [N,C,T], got {x.shape}")
    if y is not None:
        y = np.asarray(y).reshape(-1)
        try:
            y = y.astype(np.int64)
        except Exception:
            pass
    if s is not None:
        s = np.asarray(s).reshape(-1)
        try:
            s = s.astype(np.int64)
        except Exception:
            pass

    return x, y, s


def _apply_label_filter(x: np.ndarray, y: Optional[np.ndarray], s: Optional[np.ndarray],
                        label_filter: str
                        ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict]:
    stats = {"n_nonp300": None, "n_p300": None}

    if y is None or label_filter == "all":
        if y is not None:
            stats["n_nonp300"] = int((y == 0).sum())
            stats["n_p300"] = int((y == 1).sum())
        return x, y, s, stats

    if label_filter == "p300":
        keep = (y == 1)
    elif label_filter == "nonp300":
        keep = (y == 0)
    else:
        raise ValueError(f"Unknown label_filter: {label_filter!r}")

    idx = np.where(keep)[0]
    x = x[idx]
    y = y[idx]
    if s is not None:
        s = s[idx]

    stats["n_nonp300"] = int((y == 0).sum())
    stats["n_p300"] = int((y == 1).sum())
    return x, y, s, stats


def _parse_subjects_arg(arg) -> Optional[List[int]]:
    if arg is None:
        return None
    if isinstance(arg, (list, tuple)):
        return [int(s) for s in arg]
    s = str(arg).strip().lower()
    if s in ("", "all"):
        return None
    return [int(t) for t in s.replace(";", ",").split(",") if t.strip()]


def _maybe_filter_subjects(x: np.ndarray, y: Optional[np.ndarray], s: Optional[np.ndarray],
                           subjects_keep: Optional[Sequence[int]]):
    if subjects_keep is None or s is None:
        return x, y, s
    keep_set = set(int(v) for v in subjects_keep)
    keep = np.array([int(v) in keep_set for v in s])
    idx = np.where(keep)[0]
    x = x[idx]
    if y is not None:
        y = y[idx]
    s = s[idx]
    return x, y, s


def _gather_npz_paths(config) -> List[str]:
    single = (getattr(config, "speller_npz", "") or "").strip()
    if single:
        if not os.path.isfile(single):
            raise FileNotFoundError(f"speller_npz not found: {single}")
        return [single]

    multi = (getattr(config, "speller_npz_files", "") or "").strip()
    if multi:
        items: List[str] = []
        for token in multi.split(","):
            token = token.strip()
            if not token:
                continue
            matched = sorted(glob.glob(token))
            if matched:
                items.extend(matched)
            elif os.path.isfile(token):
                items.append(token)
            else:
                raise FileNotFoundError(f"speller_npz_files entry not found: {token}")
        out = sorted(set(items))
        if not out:
            raise FileNotFoundError("speller_npz_files resolved to empty set")
        return out

    folder = (getattr(config, "speller_npz_dir", "") or "").strip()
    if folder:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"speller_npz_dir not found: {folder}")
        out = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if not out:
            raise FileNotFoundError(f"No .npz under {folder}")
        return out

    raise ValueError(
        "Speller loader: must provide one of --speller_npz, "
        "--speller_npz_files, or --speller_npz_dir"
    )


def _load_one_speller_npz(path: str, split: str
                          ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict]:
    pack = np.load(path, allow_pickle=True)
    x, y, s = _select_split(pack, split)

    fs_hz = None
    if "fs" in pack.files:
        try:
            fs_hz = float(pack["fs"])
        except Exception:
            pass

    meta = {
        "path": os.path.abspath(path),
        "name": (pack["name"].item() if "name" in pack.files
                 else os.path.splitext(os.path.basename(path))[0]),
        "N": int(x.shape[0]),
        "C": int(x.shape[1]),
        "T_before": int(x.shape[-1]),
        "T_after": int(x.shape[-1]),
        "target_len": int(pack["target_len"]) if "target_len" in pack.files else int(x.shape[-1]),
        "fs_hz_in_npz": fs_hz,
        "has_labels": y is not None,
        "has_subjects": s is not None,
        "split": split,
    }
    return x, y, s, meta


def _load_and_harmonize_speller(
    paths: List[str],
    split: str,
    pad_multiple: int,
    label_filter: str,
    subjects_keep: Optional[List[int]],
):
    arrays: List[np.ndarray] = []
    labels_list: List[Optional[np.ndarray]] = []
    subjects_list: List[Optional[np.ndarray]] = []
    files_meta: List[Dict] = []

    c_ref: Optional[int] = None
    fs_ref: Optional[float] = None
    total_nonp300 = 0
    total_p300 = 0

    for path in paths:
        x, y, s, meta = _load_one_speller_npz(path, split)

        if c_ref is None:
            c_ref = x.shape[1]
        elif x.shape[1] != c_ref:
            raise ValueError(
                f"Channel mismatch: {path} has C={x.shape[1]}, expected {c_ref}"
            )

        fs_in = meta.get("fs_hz_in_npz")
        if fs_ref is None and fs_in is not None:
            fs_ref = float(fs_in)
        elif fs_ref is not None and fs_in is not None and abs(float(fs_in) - fs_ref) > 1e-6:
            raise ValueError(
                f"Sampling-rate mismatch: {path} has fs={fs_in}, expected {fs_ref}"
            )

        x, y, s = _maybe_filter_subjects(x, y, s, subjects_keep)
        x, y, s, stats = _apply_label_filter(x, y, s, label_filter)

        total_nonp300 += int(stats["n_nonp300"] or 0)
        total_p300 += int(stats["n_p300"] or 0)

        arrays.append(x.astype(np.float32))
        labels_list.append(y)
        subjects_list.append(s)

        files_meta.append({
            "path": meta["path"],
            "name": meta["name"],
            "N_raw": meta["N"],
            "N_after_filter": int(x.shape[0]),
            "C": int(x.shape[1]),
            "T_before": int(meta["T_before"]),
        })

    if not arrays:
        raise RuntimeError("No Speller arrays were loaded")

    max_t = max(arr.shape[-1] for arr in arrays)
    padded: List[np.ndarray] = []
    for arr in arrays:
        if arr.shape[-1] < max_t:
            arr = np.pad(arr, ((0, 0), (0, 0), (0, max_t - arr.shape[-1])), mode="constant")
        arr = _pad_time_to_multiple(arr, pad_multiple)
        padded.append(arr)
    arrays = padded

    x_all = np.concatenate(arrays, axis=0)

    y_all: Optional[np.ndarray] = None
    if any(y is not None for y in labels_list):
        if any(y is None for y in labels_list):
            raise ValueError(
                "Mixed labeled/unlabeled npz files; refusing to silently drop labels"
            )
        y_all = np.concatenate(labels_list, axis=0)

    s_all: Optional[np.ndarray] = None
    if any(s is not None for s in subjects_list):
        if any(s is None for s in subjects_list):
            raise ValueError(
                "Mixed npz with/without subjects field; refusing to silently drop"
            )
        s_all = np.concatenate(subjects_list, axis=0)

    valid_mask = np.ones((x_all.shape[0], x_all.shape[-1]), dtype=np.float32)

    bundle_meta = {
        "name": (f"Speller[{len(paths)}]" if len(paths) > 1
                 else os.path.splitext(os.path.basename(paths[0]))[0]),
        "files": files_meta,
        "N": int(x_all.shape[0]),
        "C": int(x_all.shape[1]),
        "T_before": int(max_t),
        "T_after": int(x_all.shape[-1]),
        "target_len": int(x_all.shape[-1]),
        "split": split,
        "fs_hz": fs_ref,
        "label_filter": label_filter,
        "n_nonp300": total_nonp300 if (total_nonp300 + total_p300) > 0 else None,
        "n_p300": total_p300 if (total_nonp300 + total_p300) > 0 else None,
        "has_labels": y_all is not None,
        "has_subjects": s_all is not None,
        "valid_ratio": float(valid_mask.sum() / valid_mask.size),
    }
    return x_all, y_all, s_all, valid_mask, bundle_meta


def data_loader_speller(config) -> Dict:
    paths = _gather_npz_paths(config)
    pad_multiple = int(getattr(config, "pad_multiple", 16))
    split = str(getattr(config, "split", "all"))
    label_filter = str(getattr(config, "label_filter", "all"))
    subjects_keep = _parse_subjects_arg(getattr(config, "subjects", "all"))
    batch_size = int(getattr(config, "batch_size", 128))
    num_workers = int(getattr(config, "num_workers", 0))

    x_all, y_all, s_all, valid_mask, bundle_meta = _load_and_harmonize_speller(
        paths=paths,
        split=split,
        pad_multiple=pad_multiple,
        label_filter=label_filter,
        subjects_keep=subjects_keep,
    )

    dataset = ArrayReconDataset(x_all)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    in_coord_path = getattr(config, "input_coord_path", None)
    tgt_coord_path = getattr(config, "target_coord_path", None)
    if not in_coord_path or not tgt_coord_path:
        raise ValueError(
            "data_loader_speller requires --input_coord_path and --target_coord_path"
        )
    input_coords = _df_to_coord_dict(pd.read_excel(in_coord_path, index_col=0))
    target_coords = _df_to_coord_dict(pd.read_excel(tgt_coord_path, index_col=0))

    if len(input_coords) != x_all.shape[1]:
        raise ValueError(
            f"input_coord_path lists {len(input_coords)} channels but data has "
            f"{x_all.shape[1]}. Provide a coord file matching the speller montage."
        )

    return {
        "loader": dataloader,
        "meta": bundle_meta,
        "input_coords": input_coords,
        "target_coords": target_coords,
        "valid_mask": valid_mask,
        "labels": y_all,
        "subjects": s_all,
        "paths": paths,
    }
