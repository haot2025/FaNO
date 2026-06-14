import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import time
import math
import argparse
from types import SimpleNamespace

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.utilities3 import MatReader, LpLoss


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class NeuralSpectralBlock2d(nn.Module):
    def __init__(self, width, num_basis, patch_size=(3, 3), num_token=4):
        super().__init__()
        self.patch_size = list(patch_size)
        self.width = width
        self.num_basis = num_basis

        modes = (1.0 / float(num_basis)) * torch.tensor(
            [i for i in range(num_basis)], dtype=torch.float
        )
        self.register_buffer("modes_list", modes)

        self.weights = nn.Parameter((1 / width) * torch.rand(width, self.num_basis * 2))
        self.head = 8
        assert width % self.head == 0, f"width={width} must be divisible by head={self.head}"
        self.num_token = num_token

        self.latent = nn.Parameter(
            (1 / width) * torch.rand(self.head, self.num_token, width // self.head)
        )
        self.encoder_attn = nn.Conv2d(self.width, self.width * 2, kernel_size=1)
        self.decoder_attn = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def self_attn(self, q, k, v):
        attn = self.softmax(torch.einsum("bhlc,bhsc->bhls", q, k))
        return torch.einsum("bhls,bhsc->bhlc", attn, v)

    def latent_encoder_attn(self, x):
        B, C, H, W = x.shape
        L = H * W
        latent_token = self.latent[None].repeat(B, 1, 1, 1)
        x_tmp = (
            self.encoder_attn(x)
            .view(B, C * 2, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, L, self.head, C // self.head, 2)
            .permute(4, 0, 2, 1, 3)
            .contiguous()
        )
        latent_token = self.self_attn(latent_token, x_tmp[0], x_tmp[1]) + latent_token
        latent_token = latent_token.permute(0, 1, 3, 2).contiguous().view(B, C, self.num_token)
        return latent_token

    def latent_decoder_attn(self, x, latent_token):
        x_init = x
        B, C, H, W = x.shape
        L = H * W
        latent_token = (
            latent_token.view(B, self.head, C // self.head, self.num_token)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        x_tmp = (
            self.decoder_attn(x)
            .view(B, C, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, L, self.head, C // self.head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x = self.self_attn(x_tmp, latent_token, latent_token)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C, H, W) + x_init
        return x

    def get_basis(self, x):
        x_sin = torch.sin(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        x_cos = torch.cos(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        return torch.cat([x_sin, x_cos], dim=-1)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bilm,im->bil", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        assert H % ph == 0 and W % pw == 0, f"H,W={H},{W} not divisible by patch={self.patch_size}"

        x = (
            x.view(B, C, H // ph, ph, W // pw, pw)
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(B * (H // ph) * (W // pw), C, ph, pw)
        )

        latent_token = self.latent_encoder_attn(x)
        latent_token_modes = self.get_basis(latent_token)
        latent_token = self.compl_mul2d(latent_token_modes, self.weights) + latent_token
        x = self.latent_decoder_attn(x, latent_token)

        x = (
            x.view(B, H // ph, W // pw, C, ph, pw)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(B, C, H, W)
        )
        return x


class NeuralSpectralUNet(nn.Module):
    def __init__(self, args, bilinear=True):
        super().__init__()
        in_channels = args.in_dim
        out_channels = args.out_dim
        width = args.d_model
        num_token = args.num_token
        num_basis = args.num_basis
        patch_size = tuple(int(x) for x in args.patch_size.split(","))
        padding = [int(x) for x in args.padding.split(",")]

        self.inc = DoubleConv(width, width)
        self.down1 = Down(width, width * 2)
        self.down2 = Down(width * 2, width * 4)
        self.down3 = Down(width * 4, width * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(width * 8, width * 16 // factor)

        self.up1 = Up(width * 16, width * 8 // factor, bilinear)
        self.up2 = Up(width * 8, width * 4 // factor, bilinear)
        self.up3 = Up(width * 4, width * 2 // factor, bilinear)
        self.up4 = Up(width * 2, width, bilinear)
        self.outc = OutConv(width, width)

        self.process1 = NeuralSpectralBlock2d(width, num_basis, patch_size, num_token)
        self.process2 = NeuralSpectralBlock2d(width * 2, num_basis, patch_size, num_token)
        self.process3 = NeuralSpectralBlock2d(width * 4, num_basis, patch_size, num_token)
        self.process4 = NeuralSpectralBlock2d(width * 8, num_basis, patch_size, num_token)
        self.process5 = NeuralSpectralBlock2d(width * 16 // factor, num_basis, patch_size, num_token)

        self.padding = padding
        self.fc0 = nn.Linear(in_channels + 2, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        if not all(item == 0 for item in self.padding):
            x = F.pad(x, [0, self.padding[0], 0, self.padding[1]])

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(self.process5(x5), self.process4(x4))
        x = self.up2(x, self.process3(x3))
        x = self.up3(x, self.process2(x2))
        x = self.up4(x, self.process1(x1))
        x = self.outc(x)

        if not all(item == 0 for item in self.padding):
            x = x[..., :-self.padding[1], :-self.padding[0]]

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/piececonst_r421_N1024_smooth1.mat")
    parser.add_argument("--test_path", default="data/piececonst_r421_N1024_smooth2.mat")
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--r", type=int, default=5)
    parser.add_argument("--s", type=int, default=85)

    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--num_token", type=int, default=4)
    parser.add_argument("--num_basis", type=int, default=12)
    parser.add_argument("--patch_size", type=str, default="3,3")
    parser.add_argument("--padding", type=str, default="11,11")

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="darcy_neuralspectral_unet_d32_ep500")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    print("args =", vars(args), flush=True)

    reader = MatReader(args.train_path)
    x_train = reader.read_field("coeff")[:args.ntrain, ::args.r, ::args.r][:, :args.s, :args.s]
    y_train = reader.read_field("sol")[:args.ntrain, ::args.r, ::args.r][:, :args.s, :args.s]

    reader = MatReader(args.test_path)
    x_test = reader.read_field("coeff")[:args.ntest, ::args.r, ::args.r][:, :args.s, :args.s]
    y_test = reader.read_field("sol")[:args.ntest, ::args.r, ::args.r][:, :args.s, :args.s]

    x_train = x_train.reshape(args.ntrain, args.s, args.s, 1)
    x_test = x_test.reshape(args.ntest, args.s, args.s, 1)

    print("x_train:", tuple(x_train.shape), "y_train:", tuple(y_train.shape))
    print("x_test:", tuple(x_test.shape), "y_test:", tuple(y_test.shape))

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    model_args = SimpleNamespace(
        in_dim=1,
        out_dim=1,
        d_model=args.d_model,
        num_token=args.num_token,
        num_basis=args.num_basis,
        patch_size=args.patch_size,
        padding=args.padding,
    )
    model = NeuralSpectralUNet(model_args).to(device)
    print("params =", model.count_params(), f"({model.count_params()/1e6:.3f}M)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    myloss = LpLoss(size_average=False)

    os.makedirs("model", exist_ok=True)
    os.makedirs("pred", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    path_model_best = f"model/{args.tag}_best.pth"
    path_model_last = f"model/{args.tag}_last.pth"
    path_pred = f"pred/{args.tag}.mat"
    path_metrics = f"results/{args.tag}_metrics.txt"

    best_test_l2 = float("inf")
    best_epoch = -1

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_l2 = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x).squeeze(-1)
            loss = myloss(out.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1))
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()
        train_l2 /= args.ntrain

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                out = model(x).squeeze(-1)
                test_l2 += myloss(out.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1)).item()

        test_l2 /= args.ntest

        if test_l2 < best_test_l2:
            best_test_l2 = test_l2
            best_epoch = ep
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "test_l2": test_l2,
                "args": vars(args),
                "params": model.count_params(),
            }, path_model_best)

        torch.save({
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "test_l2": test_l2,
            "args": vars(args),
            "params": model.count_params(),
        }, path_model_last)

        print(
            f"epoch={ep:04d} time={time.time()-t0:.3f} "
            f"train_l2={train_l2:.8e} test_l2={test_l2:.8e} "
            f"best={best_test_l2:.8e} best_epoch={best_epoch}",
            flush=True,
        )

    ckpt = torch.load(path_model_best, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pred = torch.zeros(args.ntest, args.s, args.s)
    index = 0
    sample_rel = []

    with torch.no_grad():
        for x, y in test_loader:
            bs = x.shape[0]
            x = x.to(device)
            y = y.to(device)
            out = model(x).squeeze(-1).cpu()
            pred[index:index + bs] = out

            yy = y.cpu()
            rel = torch.norm((out - yy).reshape(bs, -1), dim=1) / torch.norm(yy.reshape(bs, -1), dim=1)
            sample_rel.extend(rel.numpy().tolist())
            index += bs

    sample_rel = np.asarray(sample_rel)
    scipy.io.savemat(path_pred, {"pred": pred.numpy()})

    with open(path_metrics, "w") as f:
        f.write("model_name: NeuralSpectralUNet\n")
        f.write(f"train_path: {args.train_path}\n")
        f.write(f"test_path: {args.test_path}\n")
        f.write(f"ntrain: {args.ntrain}\n")
        f.write(f"ntest: {args.ntest}\n")
        f.write(f"r: {args.r}\n")
        f.write(f"s: {args.s}\n")
        f.write(f"d_model: {args.d_model}\n")
        f.write(f"num_token: {args.num_token}\n")
        f.write(f"num_basis: {args.num_basis}\n")
        f.write(f"patch_size: {args.patch_size}\n")
        f.write(f"padding: {args.padding}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"weight_decay: {args.weight_decay}\n")
        f.write(f"params: {model.count_params()}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_test_l2: {best_test_l2:.8e}\n")
        f.write(f"mean_rel_l2_from_pred: {sample_rel.mean():.8e}\n")
        f.write(f"median_rel_l2: {np.median(sample_rel):.8e}\n")
        f.write(f"q90_rel_l2: {np.quantile(sample_rel, 0.90):.8e}\n")
        f.write(f"q95_rel_l2: {np.quantile(sample_rel, 0.95):.8e}\n")
        f.write(f"best_rel_l2: {sample_rel.min():.8e}\n")
        f.write(f"worst_rel_l2: {sample_rel.max():.8e}\n")
        f.write(f"path_model_best: {path_model_best}\n")
        f.write(f"path_model_last: {path_model_last}\n")
        f.write(f"path_pred: {path_pred}\n")

    print("DONE")
    print("best_epoch =", best_epoch)
    print("best_test_l2 =", best_test_l2)
    print("pred saved to", path_pred)
    print("metrics saved to", path_metrics)


if __name__ == "__main__":
    main()
