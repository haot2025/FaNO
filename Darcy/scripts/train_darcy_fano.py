import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import argparse
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer

from src.data.utilities3 import MatReader, UnitGaussianNormalizer, LpLoss

def count_real_params(model):
    """Count real-valued trainable parameters.

    Complex-valued tensors are counted as two real-valued parameters.
    """
    total = 0
    for param in model.parameters():
        if param.requires_grad:
            total += 2 * param.numel() if param.is_complex() else param.numel()
    return total



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--persistent_ratio', type=float, default=0.2,
                    help='ratio of persistent/global response channels')
parser.add_argument('--modes', type=int, default=12)
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--ntest', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--r', type=int, default=5)
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

persistent_ratio = args.persistent_ratio
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
# Fourier layers
# ============================================================
class conv_fano(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, persistent_ratio=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.global_channels = max(1, int(round(out_channels * persistent_ratio)))
        self.local_channels = out_channels - self.global_channels
        assert self.local_channels > 0, "local_channels must be > 0"
        assert self.global_channels > 0, "global_channels must be > 0"

        self.scale = 1.0 / (in_channels * out_channels)

        # dynamic branch
        self.weights1 = nn.Parameter(
            torch.randn(in_channels, self.local_channels, modes1, modes2, dtype=torch.complex64) * self.scale
        )
        self.weights2 = nn.Parameter(
            torch.randn(in_channels, self.local_channels, modes1, modes2, dtype=torch.complex64) * self.scale
        )

        # persistent branch
        self.weights_x = nn.Parameter(
            torch.randn(in_channels, self.global_channels, dtype=torch.float32) * self.scale
        )
        self.weights3 = nn.Parameter(
            torch.randn(1, self.global_channels, modes1, modes2, dtype=torch.complex64) * self.scale
        )
        self.weights4 = nn.Parameter(
            torch.randn(1, self.global_channels, modes1, modes2, dtype=torch.complex64) * self.scale
        )

        # mixing operator you added back
        '''self.w1 = nn.Parameter(
            torch.randn(out_channels, out_channels, dtype=torch.complex64) * self.scale
        )'''

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.float()
        x_ft = torch.fft.rfft2(x, norm="ortho")

        weight = torch.zeros(
            self.in_channels, self.local_channels, H, W // 2 + 1,
            dtype=x_ft.dtype, device=x.device
        )
        weight[:, :, :self.modes1, :self.modes2] = self.weights1
        weight[:, :, -self.modes1:, :self.modes2] = self.weights2

        weight0 = torch.zeros(
            1, self.global_channels, H, W // 2 + 1,
            dtype=x_ft.dtype, device=x.device
        )
        weight0[:, :, :self.modes1, :self.modes2] = self.weights3
        weight0[:, :, -self.modes1:, :self.modes2] = self.weights4

        pooled = torch.mean(x, dim=[2, 3], keepdim=True)
        w_c = weight0 * torch.einsum('io,bixy->boxy', self.weights_x, pooled)

        out_ft_dynamic = torch.einsum('bchw,cohw->bohw', x_ft, weight)
        out_ft = torch.cat([out_ft_dynamic, w_c], dim=1)

        # channel mixing in Fourier space
        '''out_ft = torch.einsum('bchw,co->bohw', out_ft, self.w1)'''

        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        return x


class FaNO_block(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


# ============================================================
# FaNO network (same macro-structure as original FNO Darcy)
# ============================================================
class FaNO_Net(nn.Module):
    def __init__(self, modes1, modes2, width, s, persistent_ratio=0.2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9

        self.p = nn.Linear(3, self.width)

        self.conv0 = conv_fano(self.width, self.width, self.modes1, self.modes2, persistent_ratio)
        self.conv1 = conv_fano(self.width, self.width, self.modes1, self.modes2, persistent_ratio)
        self.conv2 = conv_fano(self.width, self.width, self.modes1, self.modes2, persistent_ratio)
        self.conv3 = conv_fano(self.width, self.width, self.modes1, self.modes2, persistent_ratio)

        self.mlp0 = FaNO_block(self.width, self.width, self.width)
        self.mlp1 = FaNO_block(self.width, self.width, self.width)
        self.mlp2 = FaNO_block(self.width, self.width, self.width)
        self.mlp3 = FaNO_block(self.width, self.width, self.width)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.q = FaNO_block(self.width, 1, self.width * 4)
        self.register_buffer("grid", self._create_grid(s, s), persistent=False)

    def _create_grid(self, size_x, size_y):
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float32).reshape(1, size_x, 1, 1)
        gridx = gridx.repeat(1, 1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float32).reshape(1, 1, size_y, 1)
        gridy = gridy.repeat(1, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

    def forward(self, x):
        grid = self.grid.repeat(x.shape[0], 1, 1, 1)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x = F.gelu(self.mlp0(self.conv0(x)) + self.w0(x))
        x = F.gelu(self.mlp1(self.conv1(x)) + self.w1(x))
        x = F.gelu(self.mlp2(self.conv2(x)) + self.w2(x))
        x = self.mlp3(self.conv3(x)) + self.w3(x)

        x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x

# ============================================================
# configs
# ============================================================
TRAIN_PATH = 'data/piececonst_r421_N1024_smooth1.mat'
TEST_PATH = 'data/piececonst_r421_N1024_smooth2.mat'

h = int(((421 - 1) / r) + 1)
s = h

iterations = epochs * (ntrain // batch_size)

ratio_tag = str(persistent_ratio).replace('.', 'p')
path = (
    f'darcy_fano_r421_N{ntrain}'
    f'_ep{epochs}'
    f'_m{modes}'
    f'_w{width}'
    f'_pr{ratio_tag}'
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
print(f'modes={modes}, width={width}, persistent_ratio={persistent_ratio}')
print(f'batch_size={batch_size}, epochs={epochs}, lr={learning_rate}')


# ============================================================
# training
# ============================================================
model = FaNO_Net(modes, modes, width, s, persistent_ratio=persistent_ratio).to(device)
print(f'params = {count_real_params(model)}')

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
            'modes': modes,
            'width': width,
            'ntrain': ntrain,
            'ntest': ntest,
            'r': r,
            's': s,
            'model_name': 'FaNO_Net',
            'persistent_ratio': persistent_ratio,
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
    'modes': modes,
    'width': width,
    'ntrain': ntrain,
    'ntest': ntest,
    'r': r,
    's': s,
    'model_name': 'FaNO_Net',
    'persistent_ratio': persistent_ratio,
}, path_model_last)

print(f'last checkpoint saved to {path_model_last}')
print(f'best epoch: {best_epoch}, best test l2: {best_test_l2}')


# ============================================================
# load best and save prediction
# ============================================================
ckpt = torch.load(path_model_best, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'loaded best checkpoint from {path_model_best}')

pred = torch.zeros(ntest, s, s)

with torch.no_grad():
    for index in range(ntest):
        x = x_test[index:index + 1]
        out = model(x).reshape(1, s, s)
        out = y_normalizer.decode(out)
        pred[index] = out.squeeze(0).cpu()

scipy.io.savemat(path_pred, mdict={'pred': pred.numpy()})
print(f'prediction saved to {path_pred}')

with open(path_metrics, 'w', encoding='utf-8') as f:
    f.write(f'path: {path}\n')
    f.write(f'model_name: FaNO_Net\n')
    f.write(f'modes: {modes}\n')
    f.write(f'width: {width}\n')
    f.write(f'persistent_ratio: {persistent_ratio}\n')
    f.write(f'ntrain: {ntrain}\n')
    f.write(f'ntest: {ntest}\n')
    f.write(f'r: {r}\n')
    f.write(f's: {s}\n')
    f.write(f'best_epoch: {best_epoch}\n')
    f.write(f'best_test_l2: {best_test_l2:.10f}\n')
    f.write(f'best_ckpt: {path_model_best}\n')
    f.write(f'last_ckpt: {path_model_last}\n')
    f.write(f'pred_path: {path_pred}\n')