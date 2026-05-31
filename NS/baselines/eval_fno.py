"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which uses a recurrent structure to propagates in time.
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)

#Complex multiplication
def compl_mul2d(a, b):
    op = partial(torch.einsum, "bctq,dctq->bdtq")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 2, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=(x.size(-2), x.size(-1)))
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(SimpleBlock2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
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

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock2d(modes, modes, width)


    def forward(self, x):
        x = self.conv1(x)
        return x


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c



################################################################
# Official FNO rollout exporter
# copied from train_fno_zongyili_m8w20_eval_fastdata.py eval protocol
################################################################

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--test_path", type=str, required=True)
parser.add_argument("--modes1", type=int, default=8)
parser.add_argument("--modes2", type=int, default=8)  # kept for command compatibility
parser.add_argument("--width", type=int, default=20)
parser.add_argument("--ntest", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--sub", type=int, default=1)
parser.add_argument("--S", type=int, default=64)
parser.add_argument("--T_in", type=int, default=10)
parser.add_argument("--T", type=int, default=10)
parser.add_argument("--step", type=int, default=1)
parser.add_argument("--split", type=str, default="last", choices=["first", "last"],
                    help="For the original 1200-sample NS file, use last 200 as test split.")
parser.add_argument("--tag", type=str, default="fno_export")
parser.add_argument("--save_npz", type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Official FNO rollout exporter")
print("Using model/eval protocol copied from train_fno_zongyili_m8w20_eval_fastdata.py")
print("args =", args)
print("=" * 60)

reader = MatReader(args.test_path)
u = reader.read_field("u")
print("raw u shape:", u.shape)

if args.split == "last":
    data = u[-args.ntest:]
else:
    data = u[:args.ntest]

test_a = data[:, ::args.sub, ::args.sub, :args.T_in]
test_u = data[:, ::args.sub, ::args.sub, args.T_in:args.T + args.T_in]

print("test_a shape:", test_a.shape)
print("test_u shape:", test_u.shape)

assert args.S == test_u.shape[-2], f"S mismatch: args.S={args.S}, data S={test_u.shape[-2]}"
assert args.T == test_u.shape[-1], f"T mismatch: args.T={args.T}, data T={test_u.shape[-1]}"

test_a = test_a.reshape(args.ntest, args.S, args.S, args.T_in)
test_u = test_u.reshape(args.ntest, args.S, args.S, args.T)

# pad the location (x,y), exactly following training script
gridx = torch.tensor(np.linspace(0, 1, args.S), dtype=torch.float)
gridx = gridx.reshape(1, args.S, 1, 1).repeat([1, 1, args.S, 1])
gridy = torch.tensor(np.linspace(0, 1, args.S), dtype=torch.float)
gridy = gridy.reshape(1, 1, args.S, 1).repeat([1, args.S, 1, 1])

test_a = torch.cat(
    (
        test_a,
        gridx.repeat([args.ntest, 1, 1, 1]),
        gridy.repeat([args.ntest, 1, 1, 1]),
    ),
    dim=-1,
)

test_a = test_a.to(device)
test_u = test_u.to(device)
gridx = gridx.to(device)
gridy = gridy.to(device)

model = Net2d(args.modes1, args.width).to(device)
ckpt = torch.load(args.ckpt, map_location=device)
state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
model.load_state_dict(state, strict=True)
model.eval()

print("loaded ckpt:", args.ckpt)
print("params:", model.count_params())

myloss = LpLoss(size_average=False)

test_l2_step = 0.0
test_l2_full = 0.0
time_step_sums = [0.0] * (args.T // args.step)

all_preds = []
all_gts = []

with torch.no_grad():
    for j in range(0, args.ntest, args.batch_size):
        bsz = min(args.batch_size, args.ntest - j)
        xx = test_a[j:j + bsz]
        yy = test_u[j:j + bsz]

        gridx_b = gridx.repeat([bsz, 1, 1, 1])
        gridy_b = gridy.repeat([bsz, 1, 1, 1])

        loss = 0.0
        pred = None

        for ti, t in enumerate(range(0, args.T, args.step)):
            y = yy[..., t:t + args.step]
            im = model(xx)

            cur = myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
            loss += cur
            time_step_sums[ti] += cur.item()

            if pred is None:
                pred = im
            else:
                pred = torch.cat((pred, im), dim=-1)

            # official autoregressive update
            xx = torch.cat(
                (
                    xx[..., args.step:-2],
                    im,
                    gridx_b,
                    gridy_b,
                ),
                dim=-1,
            )

        test_l2_step += loss.item()
        test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

        all_preds.append(pred.detach().cpu().numpy())
        all_gts.append(yy.detach().cpu().numpy())

test_step_avg = test_l2_step / args.ntest / (args.T / args.step)
test_full_avg = test_l2_full / args.ntest
per_step_np = np.array([v / args.ntest for v in time_step_sums], dtype=np.float32)

print("\nPer-step errors:")
for i, v in enumerate(per_step_np):
    print(f"t={i}: {v:.8f}")

print(f"\nZEROSHOT_RESULT tag={args.tag} S={args.S} step={test_step_avg:.8f} full={test_full_avg:.8f}")

save_path = Path(args.save_npz)
save_path.parent.mkdir(parents=True, exist_ok=True)

pred_np = np.concatenate(all_preds, axis=0).astype(np.float32)
gt_np = np.concatenate(all_gts, axis=0).astype(np.float32)

np.savez_compressed(
    save_path,
    pred=pred_np,
    gt=gt_np,
    per_step=per_step_np,
    step=np.array([test_step_avg], dtype=np.float32),
    full=np.array([test_full_avg], dtype=np.float32),
    tag=np.array([args.tag]),
    S=np.array([args.S], dtype=np.int32),
)

print(f"[OK] saved rollout npz to {save_path}")
