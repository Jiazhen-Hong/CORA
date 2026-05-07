import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import mne

from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse

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

    parser.add_argument("--input_coord_path", type=str, required=True)
    parser.add_argument("--target_coord_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--spacing", type=str, default="oct6",
                        choices=["oct4", "oct5", "oct6", "ico4", "ico5"])
    parser.add_argument("--snr", type=float, default=3.0)
    parser.add_argument("--loose", type=float, default=0.2)
    parser.add_argument("--depth", type=float, default=0.8)
    parser.add_argument("--max_dist_mm", type=float, default=20.0)

    parser.add_argument("--cov_method", type=str, default="shrunk",
                        choices=["empirical", "shrunk", "oas", "ledoit_wolf"])
    parser.add_argument("--out_dir", type=str, required=True)

    return parser


def validate_config(config):
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

    config.use_spatial_adapter = True
    config.model_name = "CORABackbone"
    config.probe_mode = "list"
    config.probe_files = ",".join(all_paths)
    config.probe_file = ""
    config.probe_dir = ""
    return config


def load_coord_df(path):
    df = pd.read_excel(path)
    required = {"name", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    return df[["name", "x", "y", "z"]].reset_index(drop=True)


def mm_to_m(xyz_mm):
    return xyz_mm / 1000.0


def build_montage(elec_df):
    ch_pos = {}
    for _, row in elec_df.iterrows():
        ch_pos[str(row["name"])] = mm_to_m(np.array([row["x"], row["y"], row["z"]], dtype=float))
    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")


def build_info(elec_df, sfreq=250):
    ch_names = elec_df["name"].astype(str).tolist()
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    montage = build_montage(elec_df)
    info.set_montage(montage, on_missing="warn")

    dummy = np.zeros((len(ch_names), int(sfreq)), dtype=float)
    raw = mne.io.RawArray(dummy, info, verbose=False)
    raw.set_eeg_reference(ref_channels="average", projection=True, verbose=False)
    return raw.info


def collect_all_trials_from_loader(loader):
    xs = []
    for xb, _ in loader:
        xs.append(xb.detach().cpu().numpy())
    X = np.concatenate(xs, axis=0)
    return X


def make_epochs_from_X(X, info):
    N, C, T = X.shape
    events = np.column_stack([
        np.arange(N),
        np.zeros(N, dtype=int),
        np.ones(N, dtype=int),
    ])
    event_id = dict(dummy=1)
    epochs = mne.EpochsArray(X, info, events=events, event_id=event_id, tmin=0.0, verbose=False)
    return epochs


def setup_source_and_fwd(info, subjects_dir, subject="fsaverage",
                         spacing="oct6", conductivity=(0.3, 0.006, 0.3)):
    print(f"[INFO] Setting up BEM and source space (spacing={spacing})")
    bem_model = mne.make_bem_model(
        subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir
    )
    bem = mne.make_bem_solution(bem_model)

    src = mne.setup_source_space(
        subject=subject, spacing=spacing, subjects_dir=subjects_dir, add_dist=False
    )

    fwd = mne.make_forward_solution(
        info, trans="fsaverage", src=src, bem=bem,
        eeg=True, meg=False, mindist=5.0, n_jobs=1, verbose=False,
    )
    return fwd, src


def get_source_space_mni_coords(src, subject, subjects_dir):
    verts_lh = src[0]["vertno"]
    verts_rh = src[1]["vertno"]
    mni_lh = mne.vertex_to_mni(verts_lh, hemis=0, subject=subject, subjects_dir=subjects_dir)
    mni_rh = mne.vertex_to_mni(verts_rh, hemis=1, subject=subject, subjects_dir=subjects_dir)
    return np.vstack([mni_lh, mni_rh])


def assign_verts_to_rois(vert_coords_mni, roi_df, max_dist_mm=20.0):
    from scipy.spatial import cKDTree
    roi_xyz = roi_df[["x", "y", "z"]].values
    tree = cKDTree(roi_xyz)
    dist, idx = tree.query(vert_coords_mni, k=1)
    vert2roi = idx.copy()
    vert2roi[dist > max_dist_mm] = -1
    print(f"[INFO] {(vert2roi >= 0).sum()}/{len(vert2roi)} vertices assigned to ROIs")
    return vert2roi


def compute_roi_sloreta_mapping(info, inv_op, lambda2, vert2roi, n_rois):
    n_elec = len(info["ch_names"])

    eye_data = np.eye(n_elec, dtype=float)
    evoked = mne.EvokedArray(eye_data, info, tmin=0.0, verbose=False)

    stc = apply_inverse(
        evoked, inv_op, lambda2=lambda2,
        method="sLORETA", pick_ori=None, verbose=False,
    )

    K = stc.data
    n_verts_total = len(vert2roi)

    if K.shape[0] == n_verts_total:
        K_src = np.abs(K)
    elif K.shape[0] % n_verts_total == 0:
        n_orient = K.shape[0] // n_verts_total
        K3 = K.reshape(n_verts_total, n_orient, n_elec)
        K_src = np.linalg.norm(K3, axis=1)
    else:
        raise RuntimeError(f"Unexpected source shape: {K.shape}")

    W = np.zeros((n_rois, n_elec), dtype=float)
    counts = np.zeros(n_rois, dtype=int)
    for v_idx, r_idx in enumerate(vert2roi):
        if r_idx < 0:
            continue
        W[r_idx] += K_src[v_idx]
        counts[r_idx] += 1

    valid = counts > 0
    W[valid] /= counts[valid][:, None]
    W[~valid] = 1.0 / n_elec

    row_sum = W.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    W = W / row_sum
    return W


def main():
    args = build_parser().parse_args()
    args = validate_config(args)
    os.makedirs(args.out_dir, exist_ok=True)

    data = data_loader(args)
    X = collect_all_trials_from_loader(data["loader"])
    print(f"[DATA] Loaded X: {X.shape}")

    elec_df = load_coord_df(args.input_coord_path)
    roi_df = load_coord_df(args.target_coord_path)

    info = build_info(elec_df, sfreq=args.fs)
    epochs = make_epochs_from_X(X, info)

    print(f"[INFO] Estimating covariance (method={args.cov_method})")
    cov = mne.compute_covariance(
        epochs, method=args.cov_method,
        tmin=None, tmax=None, rank=None, verbose=False,
    )
    cov.save(os.path.join(args.out_dir, "empirical_cov.fif"), overwrite=True)

    fs_dir = fetch_fsaverage(verbose=False)
    subjects_dir = os.path.dirname(fs_dir)
    subject = "fsaverage"

    fwd, src = setup_source_and_fwd(info, subjects_dir, subject=subject, spacing=args.spacing)

    inv_op = make_inverse_operator(
        info, fwd, cov,
        loose=args.loose, depth=args.depth, fixed=False, verbose=False,
    )
    lambda2 = 1.0 / (args.snr ** 2)

    vert_coords_mni = get_source_space_mni_coords(src, subject, subjects_dir)
    vert2roi = assign_verts_to_rois(vert_coords_mni, roi_df, max_dist_mm=args.max_dist_mm)

    W_sloreta = compute_roi_sloreta_mapping(
        info=info, inv_op=inv_op, lambda2=lambda2,
        vert2roi=vert2roi, n_rois=len(roi_df),
    )

    np.save(os.path.join(args.out_dir, "W_sloreta.npy"), W_sloreta.astype(np.float32))
    print(f"[SAVE] {os.path.join(args.out_dir, 'W_sloreta.npy')} shape={W_sloreta.shape}")

    meta = {
        "n_trials": int(X.shape[0]),
        "n_electrodes": int(X.shape[1]),
        "n_timepoints": int(X.shape[2]),
        "cov_method": args.cov_method,
        "spacing": args.spacing,
        "snr": args.snr,
        "loose": args.loose,
        "depth": args.depth,
        "max_dist_mm": args.max_dist_mm,
        "electrode_names": elec_df["name"].astype(str).tolist(),
        "roi_names": roi_df["name"].astype(str).tolist(),
    }
    with open(os.path.join(args.out_dir, "W_sloreta_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[DONE]")


if __name__ == "__main__":
    main()
