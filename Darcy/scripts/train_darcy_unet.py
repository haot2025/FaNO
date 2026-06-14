import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import numpy as np
import argparse
import numpy as np
import scipy.io
import numpy as np
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer

from src.data.utilities3 import MatReader, UnitGaussianNormalizer, LpLoss

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--modes', type=int, default=12, help='kept only for naming compatibility')
parser.add_argument('--width', type=int, default=32, help='base channel width of U-Net')
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--ntest', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--r', type=int, default=5)
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--bilinear', action='store_true', help='use bilinear upsampling instead of transposed conv')
args = parser.parse_args()

modes = args.modes
width = args.width
ntrain = args.ntrain
ntest = args.ntest
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.lr
weight_decay = args.weight_decay
r = args.r
tag = args.tag.strip()
tag_suffix = f'_{tag}' if tag else ''


# ============================================================
# U-Net blocks
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, bilinear=False):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch + skip_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, width=64, bilinear=False):
        super().__init__()
        c1 = width
        c2 = width * 2
        c3 = width * 4
        c4 = width * 8
        c5 = width * 16

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        self.up1 = Up(c5, c4, c4, bilinear=bilinear)
        self.up2 = Up(c4, c3, c3, bilinear=bilinear)
        self.up3 = Up(c3, c2, c2, bilinear=bilinear)
        self.up4 = Up(c2, c1, c1, bilinear=bilinear)
        self.outc = OutConv(c1, out_channels)

    def forward(self, x):
        # x: (B, s, s, 1)
        x = x.permute(0, 3, 1, 2)  # -> (B,1,H,W)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        x = x.permute(0, 2, 3, 1)  # -> (B,H,W,1)
        return x

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# configs
# ============================================================
TRAIN_PATH = 'data/piececonst_r421_N1024_smooth1.mat'
TEST_PATH = 'data/piececonst_r421_N1024_smooth2.mat'

h = int(((421 - 1) / r) + 1)
s = h

iterations = epochs * (ntrain // batch_size)

path = (
    f'darcy_unet_r421_N{ntrain}'
    f'_ep{epochs}'
    f'_m{modes}'
    f'_w{width}'
    f'{tag_suffix}'
)

os.makedirs('model', exist_ok=True)
os.makedirs('pred', exist_ok=True)
os.makedirs('results', exist_ok=True)

path_model_best = f'model/{path}_best.pth'
path_model_last = f'model/{path}_last.pth'
path_pred = f'pred/{path}.mat'
path_metrics = f'results/{path}_metrics.txt'


# ============================================================
# load data + normalization
# ============================================================
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')[:ntrain, ::r, ::r][:, :s, :s]
y_train = reader.read_field('sol')[:ntrain, ::r, ::r][:, :s, :s]

reader.load_file(TEST_PATH)
x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain, s, s, 1)
x_test = x_test.reshape(ntest, s, s, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
y_normalizer.cuda()

print(f'ntrain={ntrain}, ntest={ntest}, s={s}, r={r}')
print(f'width={width}, batch_size={batch_size}, epochs={epochs}, lr={learning_rate}, bilinear={args.bilinear}')


# ============================================================
# training
# ============================================================
model = UNet2d(in_channels=1, out_channels=1, width=width, bilinear=args.bilinear).to(device)
print(f'params = {model.count_params()}')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
myloss = LpLoss(size_average=False)

best_test_l2 = 1e9
best_epoch = -1

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0.0

    perm = torch.randperm(ntrain, device=device)

    for i in range(0, ntrain, batch_size):
        idx = perm[i:i + batch_size]
        x = x_train[idx]
        y = y_train[idx]
        cur_bs = x.shape[0]

        optimizer.zero_grad(set_to_none=True)

        out = model(x).reshape(cur_bs, s, s)
        out = y_normalizer.decode(out)
        y_dec = y_normalizer.decode(y)

        loss = myloss(out.reshape(cur_bs, -1), y_dec.reshape(cur_bs, -1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_l2 += loss.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for i in range(0, ntest, batch_size):
            x = x_test[i:i + batch_size]
            y = y_test[i:i + batch_size]
            cur_bs = x.shape[0]

            out = model(x).reshape(cur_bs, s, s)
            out = y_normalizer.decode(out)

            test_l2 += myloss(out.reshape(cur_bs, -1), y.reshape(cur_bs, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    if test_l2 < best_test_l2:
        best_test_l2 = test_l2
        best_epoch = ep
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_test_l2': best_test_l2,
            'width': width,
            'ntrain': ntrain,
            'ntest': ntest,
            'r': r,
            's': s,
            'model_name': 'UNet2d',
            'bilinear': args.bilinear,
        }, path_model_best)
        print(f'best checkpoint saved to {path_model_best}')

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, test_l2)

torch.save({
    'epoch': epochs - 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_test_l2': best_test_l2,
    'best_epoch': best_epoch,
    'width': width,
    'ntrain': ntrain,
    'ntest': ntest,
    'r': r,
    's': s,
    'model_name': 'UNet2d',
    'bilinear': args.bilinear,
}, path_model_last)
print(f'last checkpoint saved to {path_model_last}')


# ============================================================
# final prediction export with best ckpt
# ============================================================
ckpt = torch.load(path_model_best, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

pred = torch.zeros(ntest, s, s, device=device)
test_l2 = 0.0

with torch.no_grad():
    for i in range(0, ntest, batch_size):
        x = x_test[i:i + batch_size]
        y = y_test[i:i + batch_size]
        cur_bs = x.shape[0]

        out = model(x).reshape(cur_bs, s, s)
        out = y_normalizer.decode(out)

        pred[i:i + cur_bs] = out
        test_l2 += myloss(out.reshape(cur_bs, -1), y.reshape(cur_bs, -1)).item()

test_l2 /= ntest
pred_np = pred.detach().cpu().numpy()

scipy.io.savemat(path_pred, {'pred': pred_np})
print(f'prediction saved to {path_pred}')

with open(path_metrics, 'w', encoding='utf-8') as f:
    f.write(f'model_name: UNet2d\n')
    f.write(f'params: {model.count_params()}\n')
    f.write(f'width: {width}\n')
    f.write(f'bilinear: {args.bilinear}\n')
    f.write(f'ntrain: {ntrain}\n')
    f.write(f'ntest: {ntest}\n')
    f.write(f'r: {r}\n')
    f.write(f's: {s}\n')
    f.write(f'epochs: {epochs}\n')
    f.write(f'learning_rate: {learning_rate}\n')
    f.write(f'weight_decay: {weight_decay}\n')
    f.write(f'best_epoch: {best_epoch}\n')
    f.write(f'best_test_l2: {best_test_l2:.8e}\n')
    f.write(f'final_test_l2_from_best: {test_l2:.8e}\n')
    f.write(f'path_model_best: {path_model_best}\n')
    f.write(f'path_model_last: {path_model_last}\n')
    f.write(f'path_pred: {path_pred}\n')

print(f'metrics saved to {path_metrics}')
