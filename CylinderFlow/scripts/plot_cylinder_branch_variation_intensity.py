import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
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


def ckpt_get(args, key, default=None):
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


def build_model(ckpt_path, device):
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


def make_diffnet_input(cur_vel, node_type, stats, device):
    vel_mean = np.asarray(stats["vel_mean"], dtype=np.float32)
    vel_std = np.asarray(stats["vel_std"], dtype=np.float32)

    v = (cur_vel - vel_mean) / vel_std

    nt = node_type.reshape(-1).astype(np.int64)
    onehot = np.eye(NODE_TYPE_SIZE, dtype=np.float32)[nt]

    x = np.concatenate([v, onehot], axis=-1)
    return torch.from_numpy(x).float().unsqueeze(0).to(device)


def zero_persistent_branch(model):
    """
    Dynamic response S_E = FaNO w/o S_I.
    Persistent-branch parameters in current Cylinder FaNO implementation:
    diffusion.x_w and diffusion.w1.
    """
    zeroed = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".diffusion.x_w" in name or ".diffusion.w1" in name:
                p.zero_()
                zeroed.append(name)
    return zeroed


def zero_dynamic_branch(model):
    """
    Persistent response S_I = FaNO w/o S_E.
    Dynamic-branch parameters in current Cylinder FaNO implementation:
    diffusion.norm_w.
    """
    zeroed = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".diffusion.norm_w" in name:
                p.zero_()
                zeroed.append(name)
    return zeroed


@torch.no_grad()
def one_step(model, cur, traj, ops, t, stats, device):
    mass = torch.from_numpy(ops["mass"].astype(np.float32)).to(device)
    evecs = torch.from_numpy(ops["evecs"].astype(np.float32)).to(device)

    node_type = traj["node_type"]
    mask = loss_mask(node_type)

    delta_mean = torch.tensor(stats["delta_mean"], dtype=torch.float32, device=device)
    delta_std = torch.tensor(stats["delta_std"], dtype=torch.float32, device=device)

    x = make_diffnet_input(cur, node_type, stats, device)

    try:
        pred_norm = model(x, mass=mass, evecs=evecs)[0]
    except TypeError:
        pred_norm = model(x, mass, evecs)[0]

    pred_delta = pred_norm * delta_std + delta_mean
    pred_delta = pred_delta.detach().cpu().numpy()

    proposal = cur.copy()
    proposal[mask] = cur[mask] + pred_delta[mask]

    # 与旧 CylinderFlow rollout 口径一致：
    # 非更新节点 / 边界节点复制 GT。
    gt_next = traj["velocity"][t + 1].astype(np.float32)
    nxt = gt_next.copy()
    nxt[mask] = proposal[mask]

    return nxt.astype(np.float32)


def temporal_variation_one_series(series):
    """
    series: [T+1, V, C]
    return: [T]
    """
    diff = series[1:] - series[:-1]                 # [T,V,C]
    return np.sqrt(np.mean(diff ** 2, axis=(1, 2))) # [T]


def response_intensity_one_series(series):
    """
    series: [T+1, V, C]
    return: [T+1]

    用于 variable-size mesh 的 across-sample variation。
    每条 trajectory 在每个时间点先压成一个 mesh-size-invariant scalar。
    """
    return np.sqrt(np.mean(series ** 2, axis=(1, 2)))  # [T+1]


def aggregate_variable_mesh_metrics(series_list):
    """
    series_list: list of arrays, each [T+1,V_i,C], V_i 可变

    Returns:
        temporal_var: [T]
        sample_var: [T+1]
    """
    temporal_each = []
    intensity_each = []

    min_len = min(s.shape[0] for s in series_list)

    for s in series_list:
        s = s[:min_len]
        temporal_each.append(temporal_variation_one_series(s))
        intensity_each.append(response_intensity_one_series(s))

    temporal_each = np.stack(temporal_each, axis=0)   # [N,T]
    intensity_each = np.stack(intensity_each, axis=0) # [N,T+1]

    temporal_var = temporal_each.mean(axis=0)

    # Across-sample variation intensity:
    # std over trajectories of mesh-averaged response intensity at each rollout step.
    sample_var = intensity_each.std(axis=0)

    return temporal_var, sample_var


@torch.no_grad()
def collect_branch_series_variable_mesh(
    full_model,
    dynamic_model,
    persistent_model,
    stats,
    processed_dir,
    op_dir,
    device,
    ntest,
    max_steps,
):
    traj_paths = sorted(Path(processed_dir).glob("traj_*.npz"))[:ntest]
    if len(traj_paths) == 0:
        traj_paths = sorted(Path(processed_dir).glob("*.npz"))[:ntest]

    if len(traj_paths) == 0:
        raise RuntimeError(f"No test trajectories found in {processed_dir}")

    full_series_list = []
    dynamic_series_list = []
    persistent_series_list = []
    traj_ids = []

    for tp in tqdm(traj_paths, desc="Cylinder branch variation rollout"):
        traj = load_npz(tp)
        ops = load_npz(Path(op_dir) / f"{tp.stem}_ops.npz")

        velocity = traj["velocity"].astype(np.float32)
        cur = velocity[0].copy()
        steps = min(max_steps, velocity.shape[0] - 1)

        this_full = [cur.copy()]
        this_dynamic = [cur.copy()]
        this_persistent = [cur.copy()]

        for t in range(steps):
            nxt_full = one_step(full_model, cur, traj, ops, t, stats, device)
            nxt_dynamic = one_step(dynamic_model, cur, traj, ops, t, stats, device)
            nxt_persistent = one_step(persistent_model, cur, traj, ops, t, stats, device)

            this_full.append(nxt_full)
            this_dynamic.append(nxt_dynamic)
            this_persistent.append(nxt_persistent)

            # on-path decomposition：branch outputs 都沿 full-model rollout path 评估
            cur = nxt_full

        full_series_list.append(np.stack(this_full, axis=0))
        dynamic_series_list.append(np.stack(this_dynamic, axis=0))
        persistent_series_list.append(np.stack(this_persistent, axis=0))

        traj_ids.append(int(tp.stem.split("_")[-1]))

    return (
        np.asarray(traj_ids),
        full_series_list,
        dynamic_series_list,
        persistent_series_list,
    )


def plot_curves(steps, full_y, dynamic_y, persistent_y, ylabel, out_path):
    plt.figure(figsize=(7.5, 5.0))

    plt.plot(
        steps,
        full_y,
        color="black",
        marker="o",
        linewidth=2.2,
        markersize=4.5,
        label="Full FaNO",
    )
    plt.plot(
        steps,
        dynamic_y,
        color="red",
        marker="o",
        linewidth=2.2,
        markersize=4.5,
        label=r"Dynamic $\mathbf{S_E}$",
    )
    plt.plot(
        steps,
        persistent_y,
        color="blue",
        marker="o",
        linewidth=2.2,
        markersize=4.5,
        label=r"Persistent $\mathbf{S_I}$",
    )

    plt.xlabel("Rollout step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.28)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="results_diffnet/official_cylinder_diffnet_fano_fano_w64_k128_pr0p10/ckpt_latest.pt",
    )
    parser.add_argument("--processed_dir", default="processed_official_ar/test")
    parser.add_argument("--op_dir", default="op_cache_mgn_ar/test")
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--out_dir", default="plots/cylinder_branch_variation_intensity_pr010")
    parser.add_argument("--save_series", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_model, stats, ckpt = build_model(args.ckpt, device)

    dynamic_model = copy.deepcopy(full_model).to(device)
    persistent_model = copy.deepcopy(full_model).to(device)

    zeroed_persistent = zero_persistent_branch(dynamic_model)
    zeroed_dynamic = zero_dynamic_branch(persistent_model)

    print("=" * 100)
    print("[MODEL] Full FaNO:", args.ckpt)
    print("[BRANCH] Dynamic S_E = w/o S_I")
    print("[ZEROED persistent params]", len(zeroed_persistent))
    for x in zeroed_persistent[:20]:
        print(" ", x)

    print("[BRANCH] Persistent S_I = w/o S_E")
    print("[ZEROED dynamic params]", len(zeroed_dynamic))
    for x in zeroed_dynamic[:20]:
        print(" ", x)

    (
        traj_ids,
        full_series_list,
        dynamic_series_list,
        persistent_series_list,
    ) = collect_branch_series_variable_mesh(
        full_model=full_model,
        dynamic_model=dynamic_model,
        persistent_model=persistent_model,
        stats=stats,
        processed_dir=args.processed_dir,
        op_dir=args.op_dir,
        device=device,
        ntest=args.ntest,
        max_steps=args.max_steps,
    )

    full_time_var, full_sample_var = aggregate_variable_mesh_metrics(full_series_list)
    dynamic_time_var, dynamic_sample_var = aggregate_variable_mesh_metrics(dynamic_series_list)
    persistent_time_var, persistent_sample_var = aggregate_variable_mesh_metrics(persistent_series_list)

    steps_time = np.arange(1, len(full_time_var) + 1)
    steps_sample = np.arange(1, len(full_sample_var))  # skip initial state

    plot_curves(
        steps_time,
        full_time_var,
        dynamic_time_var,
        persistent_time_var,
        ylabel="Temporal variation intensity",
        out_path=out_dir / "cylinder_temporal_variation_intensity.png",
    )

    plot_curves(
        steps_sample,
        full_sample_var[1:],
        dynamic_sample_var[1:],
        persistent_sample_var[1:],
        ylabel="Across-sample variation intensity",
        out_path=out_dir / "cylinder_across_sample_variation_intensity.png",
    )

    rows = []
    for i, s in enumerate(steps_time):
        rows.append({
            "rollout_step": int(s),
            "full_temporal_variation": float(full_time_var[i]),
            "dynamic_temporal_variation": float(dynamic_time_var[i]),
            "persistent_temporal_variation": float(persistent_time_var[i]),
            "full_across_sample_variation": float(full_sample_var[i + 1]),
            "dynamic_across_sample_variation": float(dynamic_sample_var[i + 1]),
            "persistent_across_sample_variation": float(persistent_sample_var[i + 1]),
        })

    csv_path = out_dir / "cylinder_branch_variation_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    npz_kwargs = dict(
        traj_ids=traj_ids,
        full_time_var=full_time_var,
        dynamic_time_var=dynamic_time_var,
        persistent_time_var=persistent_time_var,
        full_sample_var=full_sample_var,
        dynamic_sample_var=dynamic_sample_var,
        persistent_sample_var=persistent_sample_var,
    )

    if args.save_series:
        # variable mesh 不能直接 np.stack，保存 object array
        npz_kwargs.update(
            full_series=np.asarray(full_series_list, dtype=object),
            dynamic_series=np.asarray(dynamic_series_list, dtype=object),
            persistent_series=np.asarray(persistent_series_list, dtype=object),
        )

    npz_path = out_dir / "cylinder_branch_variation_metrics.npz"
    np.savez(npz_path, **npz_kwargs)

    meta = {
        "ckpt": args.ckpt,
        "processed_dir": args.processed_dir,
        "op_dir": args.op_dir,
        "ntest": args.ntest,
        "max_steps": args.max_steps,
        "variable_mesh": True,
        "temporal_variation": "For each trajectory, sqrt(mean over nodes and velocity components of squared temporal difference), then averaged over trajectories.",
        "across_sample_variation": "For each trajectory and time step, first compute mesh-averaged response intensity sqrt(mean over nodes and velocity components of squared response), then take standard deviation across trajectories.",
        "dynamic_branch": "S_E = FaNO w/o S_I, persistent branch parameters x_w and w1 zeroed.",
        "persistent_branch": "S_I = FaNO w/o S_E, dynamic branch parameter norm_w zeroed.",
        "plot_style": {
            "Full FaNO": "black with marker o",
            "Dynamic S_E": "red with marker o",
            "Persistent S_I": "blue with marker o",
        },
    }

    meta_path = out_dir / "cylinder_branch_variation_metrics_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("=" * 100)
    print("[OK] saved:", out_dir / "cylinder_temporal_variation_intensity.png")
    print("[OK] saved:", out_dir / "cylinder_across_sample_variation_intensity.png")
    print("[OK] saved:", csv_path)
    print("[OK] saved:", npz_path)
    print("[OK] saved:", meta_path)

    print()
    print("[SUMMARY]")
    print("Temporal variation mean:")
    print("  Full FaNO     :", float(full_time_var.mean()))
    print("  Dynamic S_E   :", float(dynamic_time_var.mean()))
    print("  Persistent S_I:", float(persistent_time_var.mean()))
    print("Across-sample variation mean:")
    print("  Full FaNO     :", float(full_sample_var[1:].mean()))
    print("  Dynamic S_E   :", float(dynamic_sample_var[1:].mean()))
    print("  Persistent S_I:", float(persistent_sample_var[1:].mean()))


if __name__ == "__main__":
    main()
