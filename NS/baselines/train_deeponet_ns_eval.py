import argparse
from pathlib import Path
import operator
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities3 import MatReader, LpLoss


# ============================================================
# DeepONet-style model:
# Branch CNN encodes the input history u_{t-10:t}
# Trunk MLP encodes coordinates (x, y)
# Output: dot(branch, trunk) at each spatial coordinate
# ============================================================
class BranchCNN(nn.Module):
    def __init__(self, tin=10, latent_dim=128, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(tin, width, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(2),

            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(2),

            nn.Conv2d(width * 2, width * 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 4, width * 4, 3, padding=1),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(width * 4, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, latent_dim),
        )

    def forward(self, x):
        # x: [B, H, W, Tin]
        x = x.permute(0, 3, 1, 2).contiguous()
        h = self.net(x).flatten(1)
        return self.fc(h)


class TrunkMLP(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, coords):
        return self.net(coords)


class DeepONet2d(nn.Module):
    def __init__(self, tin=10, latent_dim=128, branch_width=64, trunk_hidden=128):
        super().__init__()
        self.tin = tin
        self.latent_dim = latent_dim
        self.branch = BranchCNN(tin=tin, latent_dim=latent_dim, width=branch_width)
        self.trunk = TrunkMLP(latent_dim=latent_dim, hidden_dim=trunk_hidden)
        self.bias = nn.Parameter(torch.zeros(1))

    def get_coords(self, H, W, device):
        xs = torch.linspace(0, 1, H, device=device)
        ys = torch.linspace(0, 1, W, device=device)
        gridx, gridy = torch.meshgrid(xs, ys, indexing="ij")
        coords = torch.stack([gridx.reshape(-1), gridy.reshape(-1)], dim=-1)
        return coords

    def forward(self, x):
        # x: [B, H, W, Tin]
        B, H, W, _ = x.shape
        coeff = self.branch(x)                       # [B, P]
        coords = self.get_coords(H, W, x.device)     # [HW, 2]
        basis = self.trunk(coords)                   # [HW, P]
        out = coeff @ basis.t() + self.bias          # [B, HW]
        out = out.view(B, H, W, 1)
        return out

    def count_params(self):
        return sum(reduce(operator.mul, list(p.size())) for p in self.parameters())


def load_data(path, ntrain, ntest, sub, S, T_in, T, split="last"):
    reader = MatReader(path)
    data = reader.read_field("u")
    print("raw u shape:", tuple(data.shape))

    if split == "last":
        test_data = data[-ntest:]
    else:
        test_data = data[:ntest]

    train_data = data[:ntrain]

    train_a = train_data[:, ::sub, ::sub, :T_in]
    train_u = train_data[:, ::sub, ::sub, T_in:T_in + T]

    test_a = test_data[:, ::sub, ::sub, :T_in]
    test_u = test_data[:, ::sub, ::sub, T_in:T_in + T]

    print("train_a:", tuple(train_a.shape), "train_u:", tuple(train_u.shape))
    print("test_a:", tuple(test_a.shape), "test_u:", tuple(test_u.shape))

    assert train_a.shape[1] == S and train_a.shape[2] == S, \
        f"S mismatch: expected {S}, got {train_a.shape[1:3]}"
    assert test_a.shape[1] == S and test_a.shape[2] == S, \
        f"S mismatch: expected {S}, got {test_a.shape[1:3]}"

    return train_a, train_u, test_a, test_u


def rollout_eval(model, test_a, test_u, batch_size, T, step, device, save_npz=None, tag="deeponet"):
    model.eval()
    myloss = LpLoss(size_average=False)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u),
        batch_size=batch_size,
        shuffle=False,
    )

    ntest = test_a.shape[0]
    test_l2_step = 0.0
    test_l2_full = 0.0
    time_step_sums = [0.0] * (T // step)

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for xx, yy in loader:
            xx = xx.to(device)
            yy = yy.to(device)
            bsz = xx.shape[0]

            pred = None
            loss = 0.0

            for t_idx, t in enumerate(range(0, T, step)):
                y = yy[..., t:t + step]
                im = model(xx)

                cur_loss = myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                loss += cur_loss
                time_step_sums[t_idx] += cur_loss.item()

                if pred is None:
                    pred = im
                else:
                    pred = torch.cat((pred, im), dim=-1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

            if save_npz is not None:
                all_preds.append(pred.detach().cpu().numpy())
                all_gts.append(yy.detach().cpu().numpy())

    step_avg = test_l2_step / ntest / (T / step)
    full_avg = test_l2_full / ntest
    per_step = np.array([v / ntest for v in time_step_sums], dtype=np.float32)

    if save_npz is not None:
        save_path = Path(save_npz)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            pred=np.concatenate(all_preds, axis=0).astype(np.float32),
            gt=np.concatenate(all_gts, axis=0).astype(np.float32),
            per_step=per_step,
            step=np.array([step_avg], dtype=np.float32),
            full=np.array([full_avg], dtype=np.float32),
            tag=np.array([tag]),
        )
        print(f"[OK] saved rollout npz to {save_path}")

    return step_avg, full_avg, per_step


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", choices=["train", "export"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="model/deeponet_ns1e5_ep500_best.pt")
    parser.add_argument("--save_npz", type=str, default=None)
    parser.add_argument("--tag", type=str, default="deeponet")

    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--S", type=int, default=64)
    parser.add_argument("--T_in", type=int, default=10)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--split", type=str, default="last", choices=["first", "last"])

    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.5)

    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--branch_width", type=int, default=64)
    parser.add_argument("--trunk_hidden", type=int, default=128)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("DeepONet-style NS baseline")
    print("args =", args)
    print("device =", device)
    print("=" * 80)

    if args.mode == "train":
        train_a, train_u, test_a, test_u = load_data(
            args.data_path,
            args.ntrain,
            args.ntest,
            args.sub,
            args.S,
            args.T_in,
            args.T,
            split=args.split,
        )
    else:
        # export only needs test set, but reuse loader with ntrain=min available
        train_a, train_u, test_a, test_u = load_data(
            args.data_path,
            min(args.ntrain, 1),
            args.ntest,
            args.sub,
            args.S,
            args.T_in,
            args.T,
            split=args.split,
        )

    model = DeepONet2d(
        tin=args.T_in,
        latent_dim=args.latent_dim,
        branch_width=args.branch_width,
        trunk_hidden=args.trunk_hidden,
    ).to(device)

    print("model params:", model.count_params())

    if args.mode == "export":
        ckpt = torch.load(args.ckpt, map_location=device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        print("loaded ckpt:", args.ckpt)

        step_avg, full_avg, per_step = rollout_eval(
            model,
            test_a,
            test_u,
            args.batch_size,
            args.T,
            args.step,
            device,
            save_npz=args.save_npz,
            tag=args.tag,
        )

        print("\nPer-step errors:")
        for i, v in enumerate(per_step):
            print(f"t={i}: {v:.8f}")

        print(f"\nZEROSHOT_RESULT tag={args.tag} S={args.S} step={step_avg:.8f} full={full_avg:.8f}")
        return

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
        batch_size=args.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    myloss = LpLoss(size_average=False)
    best_step = float("inf")
    best_full = float("inf")
    best_epoch = -1

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)

    print("epoch time train_l2_step train_l2_full test_l2_step test_l2_full")

    import time
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()

        train_l2_step = 0.0
        train_l2_full = 0.0

        for xx, yy in train_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            bsz = xx.shape[0]

            optimizer.zero_grad()

            pred = None
            loss = 0.0

            for t in range(0, args.T, args.step):
                y = yy[..., t:t + args.step]
                im = model(xx)

                loss = loss + myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))

                if pred is None:
                    pred = im
                else:
                    pred = torch.cat((pred, im), dim=-1)

                xx = torch.cat((xx[..., args.step:], im.detach()), dim=-1)

            loss.backward()
            optimizer.step()

            train_l2_step += loss.item()
            train_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

        scheduler.step()

        train_l2_step = train_l2_step / args.ntrain / (args.T / args.step)
        train_l2_full = train_l2_full / args.ntrain

        test_step, test_full, _ = rollout_eval(
            model,
            test_a,
            test_u,
            args.batch_size,
            args.T,
            args.step,
            device,
            save_npz=None,
            tag=args.tag,
        )

        dt = time.time() - t0
        print(f"{ep} {dt:.2f} {train_l2_step:.8f} {train_l2_full:.8f} {test_step:.8f} {test_full:.8f}", flush=True)

        if test_step < best_step:
            best_step = test_step
            best_full = test_full
            best_epoch = ep

            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_epoch": best_epoch,
                    "best_test_l2_step": best_step,
                    "best_test_l2_full": best_full,
                    "args": vars(args),
                },
                args.ckpt,
            )
            print(f"[OK] best model saved: {args.ckpt}", flush=True)

    print(f"best epoch = {best_epoch}, best test_l2_step = {best_step:.8f}, best test_l2_full = {best_full:.8f}")


if __name__ == "__main__":
    main()
