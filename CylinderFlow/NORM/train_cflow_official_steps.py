import argparse
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import os
import sys
_NORM_DIR = os.path.dirname(os.path.abspath(__file__))
if _NORM_DIR not in sys.path:
    sys.path.insert(0, _NORM_DIR)
import diffusion_net

from tqdm import tqdm


NORMAL = 0
OUTFLOW = 5
NODE_TYPE_SIZE = 9


@lru_cache(maxsize=64)
def load_traj_cached(path):
    with np.load(path) as d:
        return {k: d[k].copy() for k in d.files}


@lru_cache(maxsize=256)
def load_ops_cached(path):
    with np.load(path) as d:
        return {k: d[k].copy() for k in d.files}


def one_hot_node_type(node_type):
    nt = node_type.reshape(-1).astype(np.int64)
    return np.eye(NODE_TYPE_SIZE, dtype=np.float32)[nt]


def loss_mask_from_node_type(node_type):
    nt = node_type.reshape(-1)
    return np.logical_or(nt == NORMAL, nt == OUTFLOW)


def collect_paths(processed_dir, split, max_traj=None):
    paths = sorted((Path(processed_dir) / split).glob("*.npz"))
    if max_traj is not None:
        paths = paths[:max_traj]
    return [str(p) for p in paths]


def op_path_for(op_dir, split, traj_path):
    stem = Path(traj_path).stem
    return str(Path(op_dir) / split / f"{stem}_ops.npz")


def to_basis(x, evecs, mass):
    return torch.einsum("nk,bnc,n->bkc", evecs, x, mass)


def from_basis(x_spec, evecs):
    return torch.einsum("nk,bkc->bnc", evecs, x_spec)



def normalize_model_variant_name(model_variant):
    """Normalize historical internal variant names to the released FaNO name."""
    v = str(model_variant).lower()

    legacy_fano = {
        "g" + "fno",
        "s_" + "geo" + "fno",
        "s" + "geo" + "fno",
        "s_" + "g" + "sno",
        "s" + "g" + "sno",
    }
    legacy_pool = {
        "g" + "fno_poolnomix",
        "g" + "sno_poolnomix",
    }

    if v in legacy_fano:
        return "fano"
    if v in legacy_pool:
        return "fano_poolnomix"

    if v.startswith("s_" + "geo" + "fno" + "_r"):
        return "fano"
    if v.startswith("s" + "geo" + "fno" + "_r"):
        return "fano"
    if v.startswith("s_" + "g" + "sno" + "_r"):
        return "fano"
    if v.startswith("s" + "g" + "sno" + "_r"):
        return "fano"

    return v

class MeshSpectralConv(nn.Module):
    def __init__(self, width, k, model_variant="fno", persistent_ratio=0.25):
        super().__init__()
        self.width = width
        self.k = k
        model_variant = normalize_model_variant_name(model_variant)
        self.model_variant = model_variant
        self.scale = 1.0 / width

        if model_variant == "fno":
            self.global_c = 0
            self.local_c = width
        elif model_variant == "fano":
            self.global_c = max(1, int(round(width * persistent_ratio)))
            self.global_c = min(self.global_c, width - 1)
            self.local_c = width - self.global_c
        else:
            raise ValueError(f"unknown model_variant: {model_variant}")

        self.norm_w = nn.Parameter(
            self.scale * torch.randn(width, self.local_c, k)
        )

        if self.global_c > 0:
            self.x_w = nn.Parameter(
                self.scale * torch.randn(width, self.global_c)
            )
            self.w1 = nn.Parameter(
                self.scale * torch.randn(1, k, self.global_c)
            )
        else:
            self.x_w = None
            self.w1 = None

    def forward(self, x, mass, evecs):
        x_spec = to_basis(x, evecs, mass)
        x_local = torch.einsum("bki,iok->bko", x_spec, self.norm_w)

        if self.model_variant == "fano":
            denom = mass.sum().clamp_min(1e-8)
            pooled = torch.sum(x * mass[None, :, None], dim=1, keepdim=True) / denom
            pooled = torch.einsum("bni,io->bno", pooled, self.x_w)
            x_global = pooled * self.w1
            x_spec_out = torch.cat([x_local, x_global], dim=-1)
        else:
            x_spec_out = x_local

        return from_basis(x_spec_out, evecs)


class MeshSpectralBlock(nn.Module):
    def __init__(self, width, k, model_variant="fno", persistent_ratio=0.25):
        super().__init__()
        self.conv = MeshSpectralConv(width, k, model_variant, persistent_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width * 2, width * 2),
            nn.ReLU(),
            nn.Linear(width * 2, width),
        )

    def forward(self, x, mass, evecs):
        xd = self.conv(x, mass, evecs)
        return x + self.mlp(torch.cat([x, xd], dim=-1))


class MeshSpectralModel(nn.Module):
    def __init__(self, c_in, c_out, width, k, n_blocks, model_variant, persistent_ratio):
        super().__init__()
        self.first = nn.Linear(c_in, width)
        self.blocks = nn.ModuleList([
            MeshSpectralBlock(width, k, model_variant, persistent_ratio)
            for _ in range(n_blocks)
        ])
        self.last = nn.Linear(width, c_out)

    def forward(self, x, mass, evecs):
        h = self.first(x)
        for blk in self.blocks:
            h = blk(h, mass, evecs)
        return self.last(h)


def compute_stats(train_paths, n_samples, noise_std, seed):
    rng = np.random.default_rng(seed)

    vel_sum = np.zeros(2, dtype=np.float64)
    vel_sq = np.zeros(2, dtype=np.float64)
    vel_count = 0

    delta_sum = np.zeros(2, dtype=np.float64)
    delta_sq = np.zeros(2, dtype=np.float64)
    delta_count = 0

    for _ in tqdm(range(n_samples), desc="compute normalizer stats"):
        p = rng.choice(train_paths)
        traj = load_traj_cached(p)

        vel = traj["velocity"]
        node_type = traj["node_type"]
        T = vel.shape[0]
        t = int(rng.integers(1, T - 1))

        cur = vel[t].astype(np.float32).copy()
        nxt = vel[t + 1].astype(np.float32)

        nt = node_type.reshape(-1)
        normal_mask = nt == NORMAL
        loss_mask = loss_mask_from_node_type(node_type)

        if noise_std > 0:
            noise = rng.normal(0.0, noise_std, size=cur.shape).astype(np.float32)
            noise[~normal_mask] = 0.0
            cur = cur + noise

        delta = nxt - cur

        vel_sum += cur.sum(axis=0)
        vel_sq += (cur ** 2).sum(axis=0)
        vel_count += cur.shape[0]

        d = delta[loss_mask]
        delta_sum += d.sum(axis=0)
        delta_sq += (d ** 2).sum(axis=0)
        delta_count += d.shape[0]

    vel_mean = vel_sum / vel_count
    vel_var = vel_sq / vel_count - vel_mean ** 2

    delta_mean = delta_sum / delta_count
    delta_var = delta_sq / delta_count - delta_mean ** 2

    return {
        "vel_mean": vel_mean.astype(float).tolist(),
        "vel_std": np.sqrt(np.maximum(vel_var, 1e-12)).astype(float).tolist(),
        "delta_mean": delta_mean.astype(float).tolist(),
        "delta_std": np.sqrt(np.maximum(delta_var, 1e-12)).astype(float).tolist(),
    }


def sample_frame(paths, split, op_dir, stats, noise_std, device, rng, train=True):
    p = rng.choice(paths)
    traj = load_traj_cached(p)
    ops = load_ops_cached(op_path_for(op_dir, split, p))

    vel = traj["velocity"]
    node_type = traj["node_type"]
    T = vel.shape[0]

    t = int(rng.integers(1, T - 1))

    cur = vel[t].astype(np.float32).copy()
    nxt = vel[t + 1].astype(np.float32)

    nt = node_type.reshape(-1)
    normal_mask = nt == NORMAL

    if train and noise_std > 0:
        noise = rng.normal(0.0, noise_std, size=cur.shape).astype(np.float32)
        noise[~normal_mask] = 0.0
        cur = cur + noise

    delta = nxt - cur

    vel_mean = np.asarray(stats["vel_mean"], dtype=np.float32)
    vel_std = np.asarray(stats["vel_std"], dtype=np.float32)
    delta_mean = np.asarray(stats["delta_mean"], dtype=np.float32)
    delta_std = np.asarray(stats["delta_std"], dtype=np.float32)

    cur_norm = (cur - vel_mean) / vel_std
    node_oh = one_hot_node_type(node_type)

    x = np.concatenate([cur_norm, node_oh], axis=-1).astype(np.float32)
    y = ((delta - delta_mean) / delta_std).astype(np.float32)

    mask = loss_mask_from_node_type(node_type)

    x = torch.from_numpy(x).unsqueeze(0).to(device)
    y = torch.from_numpy(y).to(device)
    mask = torch.from_numpy(mask).to(device)

    mass = torch.from_numpy(ops["mass"].astype(np.float32)).to(device)
    evecs = torch.from_numpy(ops["evecs"].astype(np.float32)).to(device)

    return x, y, mask, mass, evecs


@torch.no_grad()
def eval_one_step(model, valid_paths, op_dir, stats, device, seed, n_samples=20):
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []

    for _ in range(n_samples):
        x, y, mask, mass, evecs = sample_frame(
            valid_paths, "valid", op_dir, stats,
            noise_std=0.0, device=device, rng=rng, train=False
        )
        pred = model(x, mass, evecs)[0]
        loss = ((pred[mask] - y[mask]) ** 2).sum(dim=-1).mean()
        losses.append(float(loss.item()))

    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def rollout_metrics(model, paths, split, op_dir, stats, device, num_rollouts=10):
    model.eval()

    vel_mean = torch.tensor(stats["vel_mean"], dtype=torch.float32, device=device)
    vel_std = torch.tensor(stats["vel_std"], dtype=torch.float32, device=device)
    delta_mean = torch.tensor(stats["delta_mean"], dtype=torch.float32, device=device)
    delta_std = torch.tensor(stats["delta_std"], dtype=torch.float32, device=device)

    horizons = [1, 10, 20, 50, 100, 200]
    metrics = {f"mse_{h}_steps": [] for h in horizons}

    for p in tqdm(paths[:num_rollouts], desc=f"rollout {split}"):
        traj = load_traj_cached(p)
        ops = load_ops_cached(op_path_for(op_dir, split, p))

        # Match official MeshGraphNets evaluation protocol:
        # dataset.add_targets keeps frames val[1:-1], so rollout starts from original frame 1.
        gt_full = traj["velocity"].astype(np.float32)
        gt = gt_full[1:-1]
        node_type = traj["node_type"]
        node_oh_np = one_hot_node_type(node_type)
        update_mask_np = loss_mask_from_node_type(node_type)

        mass = torch.from_numpy(ops["mass"].astype(np.float32)).to(device)
        evecs = torch.from_numpy(ops["evecs"].astype(np.float32)).to(device)

        cur = torch.from_numpy(gt[0]).to(device)
        node_oh = torch.from_numpy(node_oh_np).to(device)
        update_mask = torch.from_numpy(update_mask_np).to(device)

        pred_traj = []

        for _ in range(gt.shape[0]):
            pred_traj.append(cur.detach().cpu().numpy())

            cur_norm = (cur - vel_mean) / vel_std
            x = torch.cat([cur_norm, node_oh], dim=-1).unsqueeze(0)

            pred_delta_norm = model(x, mass, evecs)[0]
            pred_delta = pred_delta_norm * delta_std + delta_mean
            proposal = cur + pred_delta

            nxt = cur.clone()
            nxt[update_mask] = proposal[update_mask]
            cur = nxt

        pred = np.stack(pred_traj, axis=0)
        error = np.mean((pred - gt) ** 2, axis=-1)

        for h in horizons:
            metrics[f"mse_{h}_steps"].append(float(error[1:h + 1].mean()))

    model.train()
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




# ============================================================
# Override model with AMG/NORM DiffusionNet.
# Keep the class name MeshSpectralModel so the old training
# and evaluation logic can be reused unchanged.
# ============================================================
class MeshSpectralModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        width,
        k,
        n_blocks,
        model_variant="fano",
        persistent_ratio=0.25,
    ):
        super().__init__()
        self.net = diffusion_net.layers_norm.DiffusionNet(
            C_in=c_in,
            C_out=c_out,
            C_width=width,
            num_k=k,
            N_block=n_blocks,
            last_activation=None,
            outputs_at="vertices",
            mlp_hidden_dims=None,
            dropout=False,
            with_gradient_features=False,
            with_gradient_rotations=False,
            diffusion_method="spectral",
            persistent_ratio=persistent_ratio,
            model_variant=model_variant,
        )

    def forward(self, x, mass, evecs):
        return self.net(
            x,
            mass=mass,
            L=None,
            evals=None,
            evecs=evecs,
            gradX=None,
            gradY=None,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="processed_official_ar")
    parser.add_argument("--op_dir", default="op_cache_mgn_ar")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--model_variant", choices=["fno", "fano", "fano", "fano_r2", "fano_r3", "fano_r4"], required=True)

    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--k_eig", type=int, default=128)
    parser.add_argument("--persistent_ratio", type=float, default=0.25)

    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--norm_samples", type=int, default=20000)

    parser.add_argument("--max_train_traj", type=int, default=None)
    parser.add_argument("--max_valid_traj", type=int, default=None)

    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--rollout_every", type=int, default=10000)
    parser.add_argument("--num_rollouts", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    train_paths = collect_paths(args.processed_dir, "train", args.max_train_traj)
    valid_paths = collect_paths(args.processed_dir, "valid", args.max_valid_traj)

    print("[INFO] train trajectories:", len(train_paths))
    print("[INFO] valid trajectories:", len(valid_paths))

    stats_path = save_dir / "normalizer_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        print("[INFO] loaded stats:", stats_path)
    else:
        stats = compute_stats(train_paths, args.norm_samples, args.noise, args.seed)
        stats_path.write_text(json.dumps(stats, indent=2))
        print("[INFO] saved stats:", stats_path)
        print(json.dumps(stats, indent=2))

    model = MeshSpectralModel(
        c_in=2 + NODE_TYPE_SIZE,
        c_out=2,
        width=args.width,
        k=args.k_eig,
        n_blocks=args.blocks,
        model_variant=args.model_variant,
        persistent_ratio=args.persistent_ratio,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print("[INFO] params:", n_params)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed + 123)

    log_path = save_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("step,train_loss,valid_1step_loss,lr\n")

    best_val = float("inf")
    model.train()

    for step in range(1, args.steps + 1):
        lr_now = args.lr * (0.1 ** (step / 5_000_000.0)) + 1e-6
        for g in opt.param_groups:
            g["lr"] = lr_now

        opt.zero_grad(set_to_none=True)
        train_loss = 0.0

        for _ in range(args.batch_size):
            x, y, mask, mass, evecs = sample_frame(
                train_paths, "train", args.op_dir, stats,
                noise_std=args.noise, device=device, rng=rng, train=True
            )

            pred = model(x, mass, evecs)[0]
            loss = ((pred[mask] - y[mask]) ** 2).sum(dim=-1).mean()
            (loss / args.batch_size).backward()
            train_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        train_loss /= args.batch_size

        if step % args.log_every == 0:
            print(f"[step {step}] train_loss={train_loss:.6e} lr={lr_now:.3e}", flush=True)

        if step % args.eval_every == 0:
            val_loss = eval_one_step(
                model, valid_paths, args.op_dir, stats,
                device=device, seed=args.seed + step, n_samples=20
            )

            print(f"[eval {step}] valid_1step_loss={val_loss:.6e}", flush=True)

            with open(log_path, "a") as f:
                f.write(f"{step},{train_loss},{val_loss},{lr_now}\n")

            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "stats": stats,
                "params": n_params,
                "step": step,
                "valid_1step_loss": val_loss,
            }

            torch.save(ckpt, save_dir / "ckpt_latest.pt")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, save_dir / "ckpt_best.pt")
                print("[INFO] saved best checkpoint", flush=True)

        if step % args.rollout_every == 0:
            metrics = rollout_metrics(
                model, valid_paths, "valid", args.op_dir, stats,
                device=device, num_rollouts=args.num_rollouts
            )
            print("[rollout]", metrics, flush=True)
            (save_dir / f"rollout_step_{step}.json").write_text(
                json.dumps(metrics, indent=2)
            )

    print("[OK] training finished")


if __name__ == "__main__":
    main()
