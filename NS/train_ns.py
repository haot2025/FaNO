import os
import argparse
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


class SpectralConv2d_GSNO(nn.Module):
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


class SimpleBlock2d_GFNO_Official(nn.Module):
    def __init__(self, modes1, modes2, width, persistent_ratio=0.25, variant="base"):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.variant = variant

        # official FNO input: 10 history + x,y = 12
        self.fc0 = nn.Linear(12, self.width)

        self.conv0 = SpectralConv2d_GSNO(width, width, modes1, modes2, persistent_ratio, variant)
        self.conv1 = SpectralConv2d_GSNO(width, width, modes1, modes2, persistent_ratio, variant)
        self.conv2 = SpectralConv2d_GSNO(width, width, modes1, modes2, persistent_ratio, variant)
        self.conv3 = SpectralConv2d_GSNO(width, width, modes1, modes2, persistent_ratio, variant)

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


class Net2d_GFNO_Official(nn.Module):
    def __init__(self, modes1, modes2, width, persistent_ratio=0.25, variant="base"):
        super().__init__()
        self.conv1 = SimpleBlock2d_GFNO_Official(
            modes1, modes2, width,
            persistent_ratio=persistent_ratio,
            variant=variant
        )

    def forward(self, x):
        return self.conv1(x)

    def count_params(self):
        return sum(reduce(operator.mul, list(p.size())) for p in self.parameters())


parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, default="base",
                    choices=["base", "poolednomix", "poolednomix_postcat"])
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--persistent_ratio", type=float, default=0.25)

parser.add_argument("--train_path", type=str, default="../data/NavierStokes_V1e-5_N1200_T20.mat")
parser.add_argument("--test_path", type=str, default="../data/NavierStokes_V1e-5_N1200_T20.mat")

# strict official FNO defaults from uploaded script
parser.add_argument("--modes1", type=int, default=8)
parser.add_argument("--modes2", type=int, default=8)
parser.add_argument("--width", type=int, default=20)
parser.add_argument("--ntrain", type=int, default=1000)
parser.add_argument("--ntest", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--step_size", type=int, default=100)
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--sub", type=int, default=1)
parser.add_argument("--S", type=int, default=64)
parser.add_argument("--T_in", type=int, default=10)
parser.add_argument("--T", type=int, default=10)
parser.add_argument("--step", type=int, default=1)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tag_suffix = f"_{args.tag.strip()}" if args.tag.strip() else ""
ratio_tag = str(args.persistent_ratio).replace(".", "p")
path = (
    f"gfno_official_aligned_{args.variant}_ns_V1e-5_N{args.ntrain}"
    f"_ep{args.epochs}_m{args.modes1}x{args.modes2}_w{args.width}"
    f"_pr{ratio_tag}{tag_suffix}"
)

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

path_model_best = f"model/{path}_best.pt"
path_model_last = f"model/{path}_last.pt"
path_train_err = f"results/{path}_train.txt"
path_test_err = f"results/{path}_test.txt"

open(path_train_err, "w").close()
open(path_test_err, "w").close()

print(args.epochs, args.lr, args.step_size, args.gamma)

t0 = default_timer()

reader = MatReader(args.train_path)
train_a = reader.read_field("u")[:args.ntrain, ::args.sub, ::args.sub, :args.T_in]
train_u = reader.read_field("u")[:args.ntrain, ::args.sub, ::args.sub, args.T_in:args.T + args.T_in]

reader = MatReader(args.test_path)
test_a = reader.read_field("u")[-args.ntest:, ::args.sub, ::args.sub, :args.T_in]
test_u = reader.read_field("u")[-args.ntest:, ::args.sub, ::args.sub, args.T_in:args.T + args.T_in]

print(train_u.shape)
print(test_u.shape)

assert args.S == train_u.shape[-2]
assert args.T == train_u.shape[-1]

train_a = train_a.reshape(args.ntrain, args.S, args.S, args.T_in)
test_a = test_a.reshape(args.ntest, args.S, args.S, args.T_in)

# official training logic: append grid before DataLoader
gridx = torch.tensor(np.linspace(0, 1, args.S), dtype=torch.float32)
gridx = gridx.reshape(1, args.S, 1, 1).repeat([1, 1, args.S, 1])
gridy = torch.tensor(np.linspace(0, 1, args.S), dtype=torch.float32)
gridy = gridy.reshape(1, 1, args.S, 1).repeat([1, args.S, 1, 1])

train_a = torch.cat(
    (train_a, gridx.repeat([args.ntrain, 1, 1, 1]), gridy.repeat([args.ntrain, 1, 1, 1])),
    dim=-1
)
test_a = torch.cat(
    (test_a, gridx.repeat([args.ntest, 1, 1, 1]), gridy.repeat([args.ntest, 1, 1, 1])),
    dim=-1
)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_a, train_u),
    batch_size=args.batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_a, test_u),
    batch_size=args.batch_size,
    shuffle=False
)

print("preprocessing finished, time used:", default_timer() - t0)

gridx = gridx.to(device)
gridy = gridy.to(device)

model = Net2d_GFNO_Official(
    args.modes1, args.modes2, args.width,
    persistent_ratio=args.persistent_ratio,
    variant=args.variant
).to(device)

print(model.count_params())

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
myloss = LpLoss(size_average=False)

best_test_l2_step = float("inf")
best_test_l2_full = float("inf")
best_epoch = -1

for ep in range(args.epochs):
    model.train()
    t1 = default_timer()

    train_l2_step = 0.0
    train_l2_full = 0.0

    for xx, yy in train_loader:
        loss = 0.0
        xx = xx.to(device)
        yy = yy.to(device)
        bsz = xx.shape[0]

        for t in range(0, args.T, args.step):
            y = yy[..., t:t + args.step]
            im = model(xx)

            loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), dim=-1)

            # official rollout logic: keep grid as last two channels
            xx = torch.cat((
                xx[..., args.step:-2],
                im,
                gridx.repeat([bsz, 1, 1, 1]),
                gridy.repeat([bsz, 1, 1, 1])
            ), dim=-1)

        train_l2_step += loss.item()
        train_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # official FNO alignment: keep model in train mode during test
    # official script uses torch.no_grad() but does NOT call model.eval()
    test_l2_step = 0.0
    test_l2_full = 0.0
    time_step_sums = [0.0] * (args.T // args.step)

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

                xx = torch.cat((
                    xx[..., args.step:-2],
                    im,
                    gridx.repeat([bsz, 1, 1, 1]),
                    gridy.repeat([bsz, 1, 1, 1])
                ), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

    train_step_avg = train_l2_step / args.ntrain / (args.T / args.step)
    train_full_avg = train_l2_full / args.ntrain
    test_step_avg = test_l2_step / args.ntest / (args.T / args.step)
    test_full_avg = test_l2_full / args.ntest

    if test_step_avg < best_test_l2_step:
        best_test_l2_step = test_step_avg
        best_test_l2_full = test_full_avg
        best_epoch = ep

        torch.save({
            "epoch": ep + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "best_epoch": best_epoch,
            "best_test_l2_step": best_test_l2_step,
            "best_test_l2_full": best_test_l2_full,
        }, path_model_best)

        print(f"best model saved to {path_model_best}")

    if ep % 50 == 0 or ep == args.epochs - 1:
        print(f"\nEpoch {ep} - per-step average test error:")
        for t_idx, v in enumerate(time_step_sums):
            print(f"t={t_idx}: {v / args.ntest:.6f}")
        print(f"Average: {sum(time_step_sums) / args.ntest / (args.T / args.step):.6f}")

    scheduler.step()
    t2 = default_timer()

    print(ep, t2 - t1, train_step_avg, train_full_avg, test_step_avg, test_full_avg)

    with open(path_train_err, "a") as f:
        f.write(f"{ep} {train_step_avg:.8f} {train_full_avg:.8f}\n")

    with open(path_test_err, "a") as f:
        f.write(f"{ep} {test_step_avg:.8f} {test_full_avg:.8f}\n")

torch.save({
    "epoch": args.epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "args": vars(args),
    "best_epoch": best_epoch,
    "best_test_l2_step": best_test_l2_step,
    "best_test_l2_full": best_test_l2_full,
}, path_model_last)

print(f"last model saved to {path_model_last}")
print(f"best epoch = {best_epoch}, best test_l2_step = {best_test_l2_step}, best test_l2_full = {best_test_l2_full}")
