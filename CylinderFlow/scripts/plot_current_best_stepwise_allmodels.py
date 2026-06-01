import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NORM"))
sys.path.insert(0, str(ROOT))

from train_cflow_official_steps import MeshSpectralModel
import train_cflow_official_steps_amg_baselines as amg_base


NORMAL = 0
OUTFLOW = 5
NODE_TYPE_SIZE = 9


def load_npz(path):
    with np.load(path) as d:
        return {k: d[k].copy() for k in d.files}


def loss_mask(node_type):
    nt = node_type.reshape(-1)
    return np.logical_or(nt == NORMAL, nt == OUTFLOW)


def state3(traj, t):
    vel = traj["velocity"][t].astype(np.float32)
    prs = traj["pressure"][t].astype(np.float32)
    if prs.ndim == 1:
        prs = prs[:, None]
    return np.concatenate([vel, prs[:, :1]], axis=-1).astype(np.float32)


def cells_to_edge_index(cells):
    c = cells.astype(np.int64)
    edges = np.concatenate([
        c[:, [0, 1]], c[:, [1, 0]],
        c[:, [1, 2]], c[:, [2, 1]],
        c[:, [2, 0]], c[:, [0, 2]],
    ], axis=0)
    edges = np.unique(edges, axis=0)
    return torch.from_numpy(edges.T).long()


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


def is_diffnet_ckpt(ckpt):
    args = ckpt.get("args", {})
    return "model_variant" in args


def build_diffnet_model(ckpt, device):
    args = ckpt.get("args", {})
    model = MeshSpectralModel(
        c_in=11,
        c_out=2,
        width=int(args.get("width", 64)),
        k=int(args.get("k_eig", 128)),
        n_blocks=int(args.get("blocks", 4)),
        model_variant=args.get("model_variant", "fano"),
        persistent_ratio=float(args.get("persistent_ratio", 0.25)),
    ).to(device)
    model.load_state_dict(state_dict_from_ckpt(ckpt), strict=True)
    model.eval()
    return model


def build_baseline_model(ckpt, ckpt_path, device):
    args_dict = ckpt.get("args", {})
    model_name = args_dict.get("model_name")
    if model_name is None:
        raise ValueError(f"Baseline ckpt missing model_name: {ckpt_path}")

    ns = SimpleNamespace(**args_dict)
    ns.config_file = args_dict.get("config_file", "NORM/config.yaml")
    ns.save_dir = str(ckpt_path.parent)
    ns.k_eig = int(args_dict.get("k_eig", 128))
    ns.lr = float(args_dict.get("lr", 1e-4))
    ns.weight_decay = float(args_dict.get("weight_decay", 1e-4))
    ns.width = int(args_dict.get("width", 64))
    ns.blocks = int(args_dict.get("blocks", 4))

    model = amg_base.build_model(model_name, ns, device)
    model.load_state_dict(state_dict_from_ckpt(ckpt), strict=True)
    model.eval()
    return model


def make_diffnet_input(cur_vel, node_type, stats, device):
    vel_mean = np.asarray(stats["vel_mean"], dtype=np.float32)
    vel_std = np.asarray(stats["vel_std"], dtype=np.float32)
    v = (cur_vel - vel_mean) / vel_std

    nt = node_type.reshape(-1).astype(np.int64)
    onehot = np.eye(NODE_TYPE_SIZE, dtype=np.float32)[nt]
    x = np.concatenate([v, onehot], axis=-1)
    return torch.from_numpy(x).float().unsqueeze(0).to(device)


@torch.no_grad()
def rollout_diffnet(model, ckpt_path, ckpt, split, num_rollouts, max_steps, device):
    stats = load_stats(ckpt_path, ckpt)
    processed_dir = ROOT / "processed_official_ar" / split
    op_dir = ROOT / "op_cache_mgn_ar" / split
    trajs = sorted(processed_dir.glob("*.npz"))[:num_rollouts]

    delta_mean = torch.tensor(stats["delta_mean"], dtype=torch.float32, device=device)
    delta_std = torch.tensor(stats["delta_std"], dtype=torch.float32, device=device)

    all_mse = []

    for tp in tqdm(trajs, desc=ckpt_path.parent.name):
        traj = load_npz(tp)
        ops = load_npz(op_dir / f"{tp.stem}_ops.npz")

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
            pred_norm = model(x, mass=mass, evecs=evecs)[0]
            pred_delta = pred_norm * delta_std + delta_mean

            proposal = cur.copy()
            proposal[mask] = cur[mask] + pred_delta.detach().cpu().numpy()[mask]

            gt_next = traj["velocity"][t + 1].astype(np.float32)
            nxt = gt_next.copy()
            nxt[mask] = proposal[mask]
            cur = nxt

            curve.append(float(np.mean((cur - gt_next) ** 2)))

        all_mse.append(curve)

    return np.mean(np.asarray(all_mse), axis=0)


@torch.no_grad()
def rollout_baseline(model, ckpt_path, ckpt, split, num_rollouts, max_steps, device):
    stats = load_stats(ckpt_path, ckpt)
    processed_dir = ROOT / "processed_official_ar" / split
    trajs = sorted(processed_dir.glob("*.npz"))[:num_rollouts]

    state_mean = torch.tensor(stats["state_mean"], dtype=torch.float32, device=device)
    state_std = torch.tensor(stats["state_std"], dtype=torch.float32, device=device)
    delta_mean = torch.tensor(stats["delta_mean"], dtype=torch.float32, device=device)
    delta_std = torch.tensor(stats["delta_std"], dtype=torch.float32, device=device)

    all_mse = []

    for tp in tqdm(trajs, desc=ckpt_path.parent.name):
        traj = load_npz(tp)
        pos = torch.from_numpy(traj["mesh_pos"].astype(np.float32)).to(device)
        edge_index = cells_to_edge_index(traj["cells"]).to(device)
        mask_np = loss_mask(traj["node_type"])
        mask = torch.from_numpy(mask_np).bool().to(device)

        cur = torch.from_numpy(state3(traj, 0)).float().to(device)
        T = traj["velocity"].shape[0]
        steps = min(max_steps, T - 1)

        curve = []
        for t in range(steps):
            x = (cur - state_mean) / state_std
            data = Data(x=x, pos=pos, edge_index=edge_index)
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

            pred_norm = amg_base.forward_model(model, data)
            pred_delta = pred_norm * delta_std + delta_mean

            proposal = cur + pred_delta

            gt_next = torch.from_numpy(state3(traj, t + 1)).float().to(device)
            nxt = gt_next.clone()
            nxt[mask] = proposal[mask]
            cur = nxt

            curve.append(float(torch.mean((cur[:, :2] - gt_next[:, :2]) ** 2).detach().cpu()))

        all_mse.append(curve)

    return np.mean(np.asarray(all_mse), axis=0)


def nice_name(path, ckpt):
    name = path.parent.name
    args = ckpt.get("args", {})

    if "official_cylinder_diffnet_fno" in name:
        if path.name == "ckpt_latest.pt":
            return "FNO final"
        return "FNO best"

    if "fano_fano" in name:
        if "pr0p10" in name:
            return "FaNO pr=0.10"
        if "pr0p50" in name:
            return "FaNO pr=0.50"
        if "pr0p75" in name:
            return "FaNO pr=0.75"
        return "FaNO pr=0.25"

    if "gnot" in name.lower():
        return "GNOT-2M"

    if "deeponet" in name.lower():
        return "DeepONet-2M"

    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_rollouts", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--ymax", type=float, default=0.2)
    parser.add_argument("--out_dir", default="plots/all_current_stepwise")
    parser.add_argument("--title", default="FNO/FaNO/GNOT/DeepONet on CylinderFlow test-20")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    plt.figure(figsize=(9, 5.5))

    for c in args.ckpts:
        p = Path(c)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            print("[SKIP missing]", p)
            continue

        ckpt = torch.load(p, map_location="cpu")
        label = nice_name(p, ckpt)
        print("[PLOT]", label, p)

        if is_diffnet_ckpt(ckpt):
            model = build_diffnet_model(ckpt, device)
            y = rollout_diffnet(model, p, ckpt, args.split, args.num_rollouts, args.max_steps, device)
        else:
            model = build_baseline_model(ckpt, p, device)
            y = rollout_baseline(model, p, ckpt, args.split, args.num_rollouts, args.max_steps, device)

        xs = np.arange(1, len(y) + 1)
        plt.plot(xs, y, linewidth=2, label=label)

        val = ckpt.get("valid_1step_loss", np.nan)
        step = ckpt.get("step", -1)
        for i, mse in enumerate(y, start=1):
            rows.append((label, str(p), step, val, i, float(mse)))

    plt.xlabel("Rollout step")
    plt.ylabel("MSE")
    plt.title(args.title)
    plt.xlim(0, args.max_steps)
    if args.ymax is not None and args.ymax > 0:
        plt.ylim(0, args.ymax)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    png = out_dir / "allmodels_stepwise_ylim.png"
    csv = out_dir / "allmodels_stepwise.csv"

    plt.savefig(png, dpi=300)

    with open(csv, "w") as f:
        f.write("label,ckpt,best_step,valid_1step_loss,rollout_step,mse\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    print("[OK] saved:", png)
    print("[OK] saved:", csv)


if __name__ == "__main__":
    main()
