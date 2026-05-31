import argparse
import operator
from functools import reduce
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities3 import MatReader, LpLoss


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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
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
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True)
        out = self.final_conv(out)
        return out


class Net2dResNet(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.conv1 = ResNet(in_channels=12, out_channels=1, base_channels=base_channels)

    def forward(self, x):
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

        d3 = F.interpolate(d3, size=(H, W), mode="bilinear", align_corners=True)
        out = self.final_conv(d3)
        return out


class Net2dUNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.conv1 = UNet(in_channels=12, out_channels=1, base_channels=base_channels)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def count_params(self):
        return sum(reduce(operator.mul, list(p.size())) for p in self.parameters())


def make_grid(n, S):
    gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float32)
    gridx = gridx.reshape(1, S, 1, 1).repeat([n, 1, S, 1])

    gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float32)
    gridy = gridy.reshape(1, 1, S, 1).repeat([n, S, 1, 1])
    return gridx, gridy


def load_state(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["resnet", "unet"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--save_npz", type=str, required=True)
    parser.add_argument("--tag", type=str, default="cnn_rollout")

    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--S", type=int, default=64)
    parser.add_argument("--T_in", type=int, default=10)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--split", type=str, default="last", choices=["first", "last"])
    parser.add_argument("--base_channels", type=int, default=None)
    args = parser.parse_args()

    if args.base_channels is None:
        args.base_channels = 64 if args.model_type == "resnet" else 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("CNN rollout exporter")
    print("args =", args)
    print("=" * 70)

    reader = MatReader(args.test_path)
    data = reader.read_field("u")
    print("raw u shape:", tuple(data.shape))

    if args.split == "last":
        data = data[-args.ntest:]
    else:
        data = data[:args.ntest]

    test_a = data[:, ::args.sub, ::args.sub, :args.T_in]
    test_u = data[:, ::args.sub, ::args.sub, args.T_in:args.T_in + args.T]

    print("test_a:", tuple(test_a.shape))
    print("test_u:", tuple(test_u.shape))

    assert args.S == test_u.shape[-2], f"args.S={args.S}, data S={test_u.shape[-2]}"
    assert args.T == test_u.shape[-1], f"args.T={args.T}, data T={test_u.shape[-1]}"

    test_a = test_a.reshape(args.ntest, args.S, args.S, args.T_in)
    test_u = test_u.reshape(args.ntest, args.S, args.S, args.T)

    gridx_all, gridy_all = make_grid(args.ntest, args.S)
    test_a = torch.cat((test_a, gridx_all, gridy_all), dim=-1)

    if args.model_type == "resnet":
        model = Net2dResNet(base_channels=args.base_channels).to(device)
    else:
        model = Net2dUNet(base_channels=args.base_channels).to(device)

    load_state(model, args.ckpt, device)
    model.eval()

    print("loaded ckpt:", args.ckpt)
    print("params:", model.count_params())

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u),
        batch_size=args.batch_size,
        shuffle=False,
    )

    gridx = torch.tensor(np.linspace(0, 1, args.S), dtype=torch.float32)
    gridx = gridx.reshape(1, args.S, 1, 1).repeat([1, 1, args.S, 1]).to(device)

    gridy = torch.tensor(np.linspace(0, 1, args.S), dtype=torch.float32)
    gridy = gridy.reshape(1, 1, args.S, 1).repeat([1, args.S, 1, 1]).to(device)

    myloss = LpLoss(size_average=False)
    test_l2_step = 0.0
    test_l2_full = 0.0
    time_step_sums = [0.0] * (args.T // args.step)

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            bsz = xx.shape[0]
            loss = 0.0
            pred = None

            for t_idx, t in enumerate(range(0, args.T, args.step)):
                y = yy[..., t:t + args.step]
                im = model(xx)

                cur_loss = myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                loss += cur_loss
                time_step_sums[t_idx] += cur_loss.item()

                if pred is None:
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
    per_step_np = np.array([v / args.ntest for v in time_step_sums], dtype=np.float32)

    print("\nPer-step errors:")
    for i, v in enumerate(per_step_np):
        print(f"t={i}: {v:.8f}")

    print(f"\nZEROSHOT_RESULT tag={args.tag} S={args.S} step={test_step_avg:.8f} full={test_full_avg:.8f}")

    save_path = Path(args.save_npz)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        pred=np.concatenate(all_preds, axis=0).astype(np.float32),
        gt=np.concatenate(all_gts, axis=0).astype(np.float32),
        per_step=per_step_np,
        step=np.array([test_step_avg], dtype=np.float32),
        full=np.array([test_full_avg], dtype=np.float32),
        tag=np.array([args.tag]),
        S=np.array([args.S], dtype=np.int32),
    )

    print(f"[OK] saved rollout npz to {save_path}")


if __name__ == "__main__":
    main()
