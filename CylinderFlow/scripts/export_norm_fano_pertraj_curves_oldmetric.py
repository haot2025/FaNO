import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NORM"))
sys.path.insert(0, str(ROOT))

from train_cflow_official_steps import MeshSpectralModel


NORMAL = 0
OUTFLOW = 5
NODE_TYPE_SIZE = 9


def load_npz(path):
    with np.load(path) as d:
        return {k: d[k].copy() for k in d.files}


def loss_mask(node_type):
    nt = node_type.reshape(-1)
    return np.logical_or(nt == NORMAL, nt == OUTFLOW)


def state_dict_from_ckpt(ckpt):
    for k in ["model", "model_state", "model_state_dict", "state_dict"]:
        if k in ckpt:
            return ckpt[k]
    raise KeyError(f"No state dict key. ckpt keys={list(ckpt.keys())}")


def load_stats(ckpt_path, ckpt):
    if "stats" in ckpt:
        return ckpt["stats"]
    p = ckpt_path.parent / "normalizer_stats.json"
    if p.exists():
        return json.loads(p.read_text())
    raise FileNotFoundError(f"Cannot find stats in ckpt or {p}")


def make_diffnet_input(cur_vel, node_type, stats, device):
    vel_mean = np.asarray(stats["vel_mean"], dtype=np.float32)
    vel_std = np.asarray(stats["vel_std"], dtype=np.float32)
    v = (cur_vel - vel_mean) / vel_std

    nt = node_type.reshape(-1).astype(np.int64)
    onehot = np.eye(NODE_TYPE_SIZE, dtype=np.float32)[nt]
    x = np.concatenate([v, onehot], axis=-1)
    return torch.from_numpy(x).float().unsqueeze(0).to(device)


def ckpt_get(args, key, default=None):
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


def build_diffnet_model(ckpt_path, device):
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})

    model = MeshSpectralModel(
        c_in=11,
        c_out=2,
        width=int(ckpt_get(args, "width", 64)),
        k=int(ckpt_get(args, "k_eig", 128)),
        n_blocks=int(ckpt_get(args, "blocks", 4)),
        model_variant=ckpt_get(args, "model_variant", "fano"),
        persistent_ratio=float(ckpt_get(args, "persistent_ratio", 0.25)),
    ).to(device)

    model.load_state_dict(state_dict_from_ckpt(ckpt), strict=True)
    model.eval()

    stats = load_stats(ckpt_path, ckpt)
    return model, stats, ckpt


def infer_ckpts(stepwise_csv):
    df = pd.read_csv(stepwise_csv)

    out = {}
    for label in df["label"].astype(str).unique():
        low = label.lower()
        ckpt = df[df["label"].astype(str) == label]["ckpt"].iloc[0]

        if ("fno" in low or "norm" in low) and "NORM" not in out:
            out["NORM"] = ckpt
            out["NORM_raw_label"] = label

        if "fano" in low and "FaNO" not in out:
            out["FaNO"] = ckpt
            out["FaNO_raw_label"] = label

    if "NORM" not in out or "FaNO" not in out:
        raise RuntimeError(f"Cannot infer ckpts from labels: {df['label'].unique()}")

    print("[CKPT] NORM raw label:", out["NORM_raw_label"])
    print("[CKPT] NORM ckpt:", out["NORM"])
    print("[CKPT] FaNO raw label:", out["FaNO_raw_label"])
    print("[CKPT] FaNO ckpt:", out["FaNO"])
    return out


@torch.no_grad()
def rollout_one_traj_oldmetric(model, ckpt_path, stats, traj, ops, device, max_steps=200):
    """
    Strictly follows rollout_diffnet() in scripts/plot_current_best_stepwise_allmodels.py:
      cur = traj["velocity"][0]
      for t in range(steps):
          predict delta
          proposal[mask] = cur[mask] + pred_delta[mask]
          gt_next = traj["velocity"][t + 1]
          nxt = gt_next.copy()
          nxt[mask] = proposal[mask]
          cur = nxt
          mse = mean((cur - gt_next)^2) over all nodes and both velocity channels
    """
    delta_mean = torch.tensor(stats["delta_mean"], dtype=torch.float32, device=device)
    delta_std = torch.tensor(stats["delta_std"], dtype=torch.float32, device=device)

    mass = torch.from_numpy(ops["mass"].astype(np.float32)).to(device)
    evecs = torch.from_numpy(ops["evecs"].astype(np.float32)).to(device)

    node_type = traj["node_type"]
    mask = loss_mask(node_type)

    cur = traj["velocity"][0].astype(np.float32)
    T = traj["velocity"].shape[0]
    steps = min(max_steps, T - 1)

    curve = []
    for t in range(steps):
        x = make_diffnet_input(cur, node_type, stats, device)

        # same as old script: model(x, mass=mass, evecs=evecs)
        pred_norm = model(x, mass=mass, evecs=evecs)[0]
        pred_delta = pred_norm * delta_std + delta_mean

        proposal = cur.copy()
        proposal[mask] = cur[mask] + pred_delta.detach().cpu().numpy()[mask]

        gt_next = traj["velocity"][t + 1].astype(np.float32)

        nxt = gt_next.copy()
        nxt[mask] = proposal[mask]
        cur = nxt

        curve.append(float(np.mean((cur - gt_next) ** 2)))

    return np.asarray(curve, dtype=np.float32)


def main():
    stepwise_csv = Path("plots/eval_fno_fano_diffu_latest_test100/allmodels_stepwise.csv")
    processed_dir = ROOT / "processed_official_ar" / "test"
    op_dir = ROOT / "op_cache_mgn_ar" / "test"
    out_dir = ROOT / "plots/eval_fno_fano_diffu_latest_test100"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_steps = 200
    ntest = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = infer_ckpts(stepwise_csv)

    labels = ["NORM", "FaNO"]
    models = {}
    stats = {}
    for label in labels:
        model, stat, _ = build_diffnet_model(ckpts[label], device)
        models[label] = model
        stats[label] = stat

    traj_paths = sorted(processed_dir.glob("*.npz"))[:ntest]
    traj_ids = [int(p.stem.split("_")[-1]) for p in traj_paths]

    all_curves = []

    for label in labels:
        print("=" * 100)
        print("[EVAL]", label)

        curves = []
        for tp in tqdm(traj_paths, desc=f"{label} oldmetric per-traj"):
            tid = int(tp.stem.split("_")[-1])
            traj = load_npz(tp)
            ops = load_npz(op_dir / f"{tp.stem}_ops.npz")

            curve = rollout_one_traj_oldmetric(
                model=models[label],
                ckpt_path=Path(ckpts[label]),
                stats=stats[label],
                traj=traj,
                ops=ops,
                device=device,
                max_steps=max_steps,
            )
            curves.append(curve)

        curves = np.stack(curves, axis=0)
        all_curves.append(curves)

        print("[CHECK]", label)
        print("shape:", curves.shape)
        print("mean MSE@1/50/100/200:",
              curves[:, 0].mean(),
              curves[:, 49].mean(),
              curves[:, 99].mean(),
              curves[:, 199].mean())

    all_curves = np.stack(all_curves, axis=0)  # [2, 100, 200]
    mean_curves = all_curves.mean(axis=1)

    out_npz = out_dir / "norm_fano_pertraj_curves_test100_oldmetric.npz"
    np.savez(
        out_npz,
        steps=np.arange(1, max_steps + 1),
        labels=np.asarray(labels),
        traj_ids=np.asarray(traj_ids),
        per_traj_curves=all_curves,
        mean_curves=mean_curves,
        ckpt_paths=np.asarray([ckpts[x] for x in labels]),
        max_steps=max_steps,
        ntest=ntest,
        metric="old_allmodels_stepwise_metric",
    )

    rows = []
    for mi, label in enumerate(labels):
        for si, tid in enumerate(traj_ids):
            for step in range(1, max_steps + 1):
                rows.append({
                    "label": label,
                    "traj_id": tid,
                    "rollout_step": step,
                    "mse": float(all_curves[mi, si, step - 1]),
                })

    out_csv = out_dir / "norm_fano_pertraj_curves_test100_oldmetric.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_json = out_dir / "norm_fano_pertraj_curves_test100_oldmetric_meta.json"
    out_json.write_text(json.dumps({
        "labels": labels,
        "ckpt_paths": {k: ckpts[k] for k in labels},
        "shape": list(all_curves.shape),
        "meaning": "per_traj_curves[model, sample, step]",
        "steps": "1..200",
        "metric": "strictly follows scripts/plot_current_best_stepwise_allmodels.py rollout_diffnet: mean((cur - gt_next)^2) over all nodes and velocity channels",
    }, indent=2))

    print("[OK] saved:", out_npz)
    print("[OK] saved:", out_csv)
    print("[OK] saved:", out_json)

    print("=" * 100)
    print("[VERIFY AGAINST allmodels_stepwise.csv]")
    old = pd.read_csv(stepwise_csv)
    for label_out, raw_label in [("NORM", ckpts["NORM_raw_label"]), ("FaNO", ckpts["FaNO_raw_label"])]:
        sub = old[old["label"].astype(str) == raw_label]
        print(label_out, "raw label:", raw_label)
        for h in [1, 50, 100, 200]:
            old_v = float(sub[sub["rollout_step"] == h]["mse"].iloc[0])
            new_v = float(mean_curves[labels.index(label_out), h - 1])
            print(f"  MSE@{h}: old={old_v:.8g} new={new_v:.8g} diff={new_v-old_v:+.3e}")


if __name__ == "__main__":
    main()
