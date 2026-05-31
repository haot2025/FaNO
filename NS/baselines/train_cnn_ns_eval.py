import os
import argparse
import operator
from functools import reduce
from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities3 import MatReader, LpLoss


torch.manual_seed(0)
np.random.seed(0)


# ============================================================
# ResNet
# ============================================================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(residual)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=1, base_channels=64, num_blocks=(2, 2, 2)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.layer1 = self._make_layer(base_channels, base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, num_blocks[2], stride=2)

        self.upsample1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1)
        self.upsample2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1)

        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.upsample1(out)
        out = self.upsample2(out)

        # 防止奇偶尺寸或者未来分辨率变化时尺寸不完全一致
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True)

        out = self.final_conv(out)
        return out


class Net2dResNet(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.conv1 = ResNet(in_channels=12, out_channels=1, base_channels=base_channels)

    def forward(self, x):
        # x: [B, S, S, 12]
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def count_params(self):
        return sum(reduce(operator.mul, list(p.size())) for p in self.parameters())


# ============================================================
# U-Net
# ============================================================
class UNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=1, base_channels=32):
        super().__init__()

        self.enc1 = self._block(in_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.pool1 = nn.MaxPool2d(2)

        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.enc4 = self._block(base_channels * 4, base_channels * 8)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = self._block(base_channels * 8, base_channels * 16)

        self.up1 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)
        self.dec1 = self._block(base_channels * 16, base_channels * 8)

        self.up2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec2 = self._block(base_channels * 8, base_channels * 4)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec3 = self._block(base_channels * 4, base_channels * 2)

        self.final_conv = nn.Conv2d(base_channels * 2, out_channels, 1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _resize_like(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=True)
        return x

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool1(e2))
        e4 = self.enc4(self.pool2(e3))

        b = self.bottleneck(self.pool2(e4))

        d1 = self.up1(b)
        d1 = self._resize_like(d1, e4)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = self._resize_like(d2, e3)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = self._resize_like(d3, e2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        # 旧脚本这里是 size=64；新脚本改成输入分辨率
        d3 = F.interpolate(d3, size=(H, W), mode="bilinear", align_corners=True)
        out = self.final_conv(d3)
        return out


class Net2dUNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.conv1 = UNet(in_channels=12, out_channels=1, base_channels=base_channels)

    def forward(self, x):
        # x: [B, S, S, 12]
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def count_params(self):
        return sum(reduce(operator.mul, list(p.size())) for p in self.parameters())


# ============================================================
# Data
# ============================================================
def make_grid(n, S):
    gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float32)
    gridx = gridx.reshape(1, S, 1, 1).repeat([n, 1, S, 1])

    gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float32)
    gridy = gridy.reshape(1, 1, S, 1).repeat([n, S, 1, 1])
    return gridx, gridy


def load_ns_data(args):
    reader = MatReader(args.data_path)
    data = reader.read_field("u")

    train_a = data[:args.ntrain, ::args.sub, ::args.sub, :args.T_in]
    train_u = data[:args.ntrain, ::args.sub, ::args.sub, args.T_in:args.T_in + args.T]

    test_a = data[-args.ntest:, ::args.sub, ::args.sub, :args.T_in]
    test_u = data[-args.ntest:, ::args.sub, ::args.sub, args.T_in:args.T_in + args.T]

    print("raw data shape:", tuple(data.shape))
    print("train_a:", tuple(train_a.shape), "train_u:", tuple(train_u.shape))
    print("test_a :", tuple(test_a.shape), "test_u :", tuple(test_u.shape))

    assert args.S == train_u.shape[-2], f"args.S={args.S}, data S={train_u.shape[-2]}"
    assert args.T == train_u.shape[-1], f"args.T={args.T}, data T={train_u.shape[-1]}"

    train_a = train_a.reshape(args.ntrain, args.S, args.S, args.T_in)
    test_a = test_a.reshape(args.ntest, args.S, args.S, args.T_in)

    train_gridx, train_gridy = make_grid(args.ntrain, args.S)
    test_gridx, test_gridy = make_grid(args.ntest, args.S)

    # 关键：历史场在前，grid 在最后；这样 xx[..., step:-2] 才是正确的历史滑窗
    train_a = torch.cat((train_a, train_gridx, train_gridy), dim=-1)
    test_a = torch.cat((test_a, test_gridx, test_gridy), dim=-1)

    return train_a, train_u, test_a, test_u


# ============================================================
# Train / Eval
# ============================================================
def evaluate(model, test_loader, myloss, gridx, gridy, args, device):
    model.eval()

    test_l2_step = 0.0
    test_l2_full = 0.0
    time_step_sums = [0.0] * (args.T // args.step)

    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            bsz = xx.shape[0]

            loss = 0.0

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

    test_step_avg = test_l2_step / args.ntest / (args.T / args.step)
    test_full_avg = test_l2_full / args.ntest

    return test_step_avg, test_full_avg, time_step_sums


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["resnet", "unet"])
    parser.add_argument("--data_path", type=str, default="../data/NavierStokes_V1e-5_N1200_T20.mat")
    parser.add_argument("--tag", type=str, default=None)

    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--S", type=int, default=64)
    parser.add_argument("--T_in", type=int, default=10)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--step", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler_step", type=int, default=100)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)

    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    if args.base_channels is None:
        args.base_channels = 64 if args.model_type == "resnet" else 32

    if args.tag is None:
        args.tag = f"{args.model_type}_ns_V1e-5_N{args.ntrain}_ep{args.epochs}_S{args.S}_bc{args.base_channels}_eval"

    os.makedirs("model", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    path_model_best = f"model/{args.tag}_best.pt"
    path_model_last = f"model/{args.tag}_last.pt"
    path_train_err = f"results/{args.tag}_train.txt"
    path_test_err = f"results/{args.tag}_test.txt"

    print("=" * 80)
    print("CNN NS training")
    print("args =", args)
    print("=" * 80)

    t0 = default_timer()
    train_a, train_u, test_a, test_u = load_ns_data(args)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("preprocessing finished, time used:", default_timer() - t0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "resnet":
        model = Net2dResNet(base_channels=args.base_channels).to(device)
    else:
        model = Net2dUNet(base_channels=args.base_channels).to(device)

    print("model params:", model.count_params())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step,
        gamma=args.scheduler_gamma,
    )

    myloss = LpLoss(size_average=False)

    gridx = torch.tensor(np.linspace(0, 1, args.S), dtype=torch.float32)
    gridx = gridx.reshape(1, args.S, 1, 1).repeat([1, 1, args.S, 1]).to(device)

    gridy = torch.tensor(np.linspace(0, 1, args.S), dtype=torch.float32)
    gridy = gridy.reshape(1, 1, args.S, 1).repeat([1, args.S, 1, 1]).to(device)

    best_test_l2_step = float("inf")
    best_test_l2_full = float("inf")
    best_epoch = -1

    train_log = []
    test_log = []

    for ep in range(args.epochs):
        model.train()
        t1 = default_timer()

        train_l2_step = 0.0
        train_l2_full = 0.0

        for xx, yy in train_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            bsz = xx.shape[0]

            loss = 0.0

            for t in range(0, args.T, args.step):
                y = yy[..., t:t + args.step]
                im = model(xx)

                loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))

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

            train_l2_step += loss.item()
            train_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_step_avg = train_l2_step / args.ntrain / (args.T / args.step)
        train_full_avg = train_l2_full / args.ntrain

        test_step_avg, test_full_avg, time_step_sums = evaluate(
            model, test_loader, myloss, gridx, gridy, args, device
        )

        if test_step_avg < best_test_l2_step:
            best_test_l2_step = test_step_avg
            best_test_l2_full = test_full_avg
            best_epoch = ep
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "best_test_l2_step": best_test_l2_step,
                    "best_test_l2_full": best_test_l2_full,
                },
                path_model_best,
            )
            print(f"best model saved to {path_model_best}")

        scheduler.step()
        t2 = default_timer()

        train_log.append([ep, train_step_avg, train_full_avg])
        test_log.append([ep, test_step_avg, test_full_avg])

        if ep % 50 == 0 or ep == args.epochs - 1:
            print(f"\nEpoch {ep} - per-step average test error:")
            for t_idx, v in enumerate(time_step_sums):
                print(f"t={t_idx}: {v / args.ntest:.6f}")
            print(f"Average: {sum(time_step_sums) / args.ntest / (args.T / args.step):.6f}")

        print(ep, t2 - t1, train_step_avg, train_full_avg, test_step_avg, test_full_avg)

        np.savetxt(path_train_err, np.array(train_log), fmt="%.10f", header="epoch train_l2_step train_l2_full")
        np.savetxt(path_test_err, np.array(test_log), fmt="%.10f", header="epoch test_l2_step test_l2_full")

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "best_epoch": best_epoch,
            "best_test_l2_step": best_test_l2_step,
            "best_test_l2_full": best_test_l2_full,
        },
        path_model_last,
    )

    print(f"last model saved to {path_model_last}")
    print(f"best epoch = {best_epoch}, best test_l2_step = {best_test_l2_step}, best test_l2_full = {best_test_l2_full}")


if __name__ == "__main__":
    main()
