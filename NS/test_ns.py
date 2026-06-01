"""
FaNO implementation for Navier-Stokes autoregressive prediction.

The core modification is the FaNO spectral block, which factorizes the
Fourier response into a dynamic branch and a persistent branch. The dynamic
branch applies input-dependent spectral filters, while the persistent branch
uses pooled global information to produce stable low-frequency responses.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
from functools import reduce
import operator

from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)


class SpectralConv2d_FaNO(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2,
                 persistent_ratio=0.25, variant="base"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.variant = variant

        self.scale = 1 / (in_channels * out_channels)

        self.persistent_channels = max(1, int(round(out_channels * persistent_ratio)))
        self.dynamic_channels = out_channels - self.persistent_channels
        if self.dynamic_channels <= 0:
            raise ValueError(f"persistent_ratio={persistent_ratio} too large")

        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, self.dynamic_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, self.dynamic_channels, modes1, modes2, dtype=torch.cfloat)
        )

        self.weights3 = nn.Parameter(
            self.scale * torch.rand(1, self.persistent_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(1, self.persistent_channels, modes1, modes2, dtype=torch.cfloat)
        )

        if variant in ["poolednomix", "poolednomix_postcat"]:
            self.persistent_gain = nn.Parameter(
                self.scale * torch.rand(1, self.persistent_channels, 1, 1, dtype=torch.float32)
            )
        else:
            self.weights_x = nn.Parameter(
                self.scale * torch.rand(in_channels, self.persistent_channels, dtype=torch.float32)
            )

        if variant == "poolednomix_postcat":
            self.post_mix = nn.Conv2d(out_channels, out_channels, 1)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft_dynamic = torch.zeros(
            B, self.dynamic_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft_dynamic[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft_dynamic[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        weight0 = torch.zeros(
            1, self.persistent_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        weight0[:, :, :self.modes1, :self.modes2] = self.weights3
        weight0[:, :, -self.modes1:, :self.modes2] = self.weights4

        pooled = torch.mean(x, dim=[2, 3], keepdim=True)

        if self.variant in ["poolednomix", "poolednomix_postcat"]:
            pooled_scalar = pooled.mean(dim=1, keepdim=True)
            persistent_coef = pooled_scalar * self.persistent_gain
        else:
            persistent_coef = torch.einsum("io,bixy->boxy", self.weights_x, pooled)

        out_ft_persistent = weight0 * persistent_coef

        out_ft = torch.cat([out_ft_dynamic, out_ft_persistent], dim=1)
        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")

        if self.variant == "poolednomix_postcat":
            x = self.post_mix(x)

        return x


class FaNO_block(nn.Module):
    def __init__(self, modes1, modes2, width, persistent_ratio=0.25, variant="base"):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.variant = variant

        # official FNO input: 10 history + x,y = 12
        self.fc0 = nn.Linear(12, self.width)

        self.conv0 = SpectralConv2d_FaNO(width, width, modes1, modes2, persistent_ratio, variant)
        self.conv1 = SpectralConv2d_FaNO(width, width, modes1, modes2, persistent_ratio, variant)
        self.conv2 = SpectralConv2d_FaNO(width, width, modes1, modes2, persistent_ratio, variant)
        self.conv3 = SpectralConv2d_FaNO(width, width, modes1, modes2, persistent_ratio, variant)

        # official FNO skip: Conv1d over flattened spatial dimension
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)

        # official FNO uses BN
        self.bn0 = nn.BatchNorm2d(width)
        self.bn1 = nn.BatchNorm2d(width)
        self.bn2 = nn.BatchNorm2d(width)
        self.bn3 = nn.BatchNorm2d(width)

        # official output head: fc1 -> ReLU -> fc2
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: [B, S, S, 12], already contains grid
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn0(x1 + x2)
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn1(x1 + x2)
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn2(x1 + x2)
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn3(x1 + x2)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class FaNO(nn.Module):
    def __init__(self, modes1, modes2, width, persistent_ratio=0.25, variant="base"):
        super().__init__()
        self.conv1 = FaNO_block(
            modes1, modes2, width,
            persistent_ratio=persistent_ratio,
            variant=variant
        )

    def forward(self, x):
        return self.conv1(x)

    def count_params(self):
        return sum(reduce(operator.mul, list(p.size())) for p in self.parameters())



parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--test_path", type=str, required=True)

parser.add_argument("--variant", type=str, default="poolednomix",
                    choices=["base", "poolednomix", "poolednomix_postcat"])
parser.add_argument("--persistent_ratio", type=float, default=0.25)

parser.add_argument("--modes1", type=int, default=8)
parser.add_argument("--modes2", type=int, default=8)
parser.add_argument("--width", type=int, default=20)

parser.add_argument("--ntest", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--sub", type=int, default=1)
parser.add_argument("--S", type=int, required=True)

parser.add_argument("--T_in", type=int, default=10)
parser.add_argument("--T", type=int, default=10)
parser.add_argument("--step", type=int, default=1)

parser.add_argument("--tag", type=str, default="zeroshot")
parser.add_argument("--save_npz", type=str, default=None, help="optional path to save rollout pred/gt as .npz")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("============================================================")
print("FaNO zero-shot resolution evaluation")
print("embedded FaNO model definitions")
print("args =", args)
print("============================================================")

reader = MatReader(args.test_path)
data = reader.read_field("u")
print("raw u shape:", data.shape)

test_a = data[-args.ntest:, ::args.sub, ::args.sub, :args.T_in]
test_u = data[-args.ntest:, ::args.sub, ::args.sub, args.T_in:args.T + args.T_in]

print("test_a shape:", test_a.shape)
print("test_u shape:", test_u.shape)

assert args.S == test_u.shape[-2], f"args.S={args.S}, data S={test_u.shape[-2]}"
assert args.T == test_u.shape[-1], f"args.T={args.T}, data T={test_u.shape[-1]}"

test_a = test_a.reshape(args.ntest, args.S, args.S, args.T_in)

gridx = torch.tensor(np.linspace(0, 1, args.S), dtype=torch.float32)
gridx = gridx.reshape(1, args.S, 1, 1).repeat([1, 1, args.S, 1])
gridy = torch.tensor(np.linspace(0, 1, args.S), dtype=torch.float32)
gridy = gridy.reshape(1, 1, args.S, 1).repeat([1, args.S, 1, 1])

test_a = torch.cat(
    (
        test_a,
        gridx.repeat([args.ntest, 1, 1, 1]),
        gridy.repeat([args.ntest, 1, 1, 1]),
    ),
    dim=-1,
)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_a, test_u),
    batch_size=args.batch_size,
    shuffle=False,
)

gridx = gridx.to(device)
gridy = gridy.to(device)

model = FaNO(
    args.modes1,
    args.modes2,
    args.width,
    persistent_ratio=args.persistent_ratio,
    variant=args.variant,
).to(device)

ckpt = torch.load(args.ckpt, map_location=device)
state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
model.load_state_dict(state, strict=True)

print("loaded ckpt:", args.ckpt)
print("params:", model.count_params())

model.eval()
myloss = LpLoss(size_average=False)

test_l2_step = 0.0
test_l2_full = 0.0
time_step_sums = [0.0] * (args.T // args.step)

all_preds = []
all_gts = []

with torch.no_grad():
    for xx, yy in test_loader:
        loss = 0.0
        xx = xx.to(device)
        yy = yy.to(device)
        bsz = xx.shape[0]

        for t_idx, t in enumerate(range(0, args.T, args.step)):
            y = yy[..., t:t + args.step]
            im = model(xx)

            cur_loss = myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
            loss += cur_loss
            time_step_sums[t_idx] += cur_loss.item()

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), dim=-1)

            xx = torch.cat(
                (
                    xx[..., args.step:-2],
                    im,
                    gridx.repeat([bsz, 1, 1, 1]),
                    gridy.repeat([bsz, 1, 1, 1]),
                ),
                dim=-1,
            )

        test_l2_step += loss.item()
        test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

        all_preds.append(pred.detach().cpu().numpy())
        all_gts.append(yy.detach().cpu().numpy())

test_step_avg = test_l2_step / args.ntest / (args.T / args.step)
test_full_avg = test_l2_full / args.ntest

print("\nZero-shot per-step errors:")
for t_idx, v in enumerate(time_step_sums):
    print(f"t={t_idx}: {v / args.ntest:.8f}")

print(f"\nZEROSHOT_RESULT tag={args.tag} S={args.S} step={test_step_avg:.8f} full={test_full_avg:.8f}")

if args.save_npz is not None:
    save_path = Path(args.save_npz)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pred_np = np.concatenate(all_preds, axis=0)
    gt_np = np.concatenate(all_gts, axis=0)
    per_step_np = np.array([v / args.ntest for v in time_step_sums], dtype=np.float32)

    np.savez_compressed(
        save_path,
        pred=pred_np.astype(np.float32),
        gt=gt_np.astype(np.float32),
        per_step=per_step_np,
        step=np.array([test_step_avg], dtype=np.float32),
        full=np.array([test_full_avg], dtype=np.float32),
        tag=np.array([args.tag]),
        S=np.array([args.S], dtype=np.int32),
    )
    print(f"[OK] saved rollout npz to {save_path}")
