import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import argparse
"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import torch.nn.functional as F
from timeit import default_timer
from src.data.utilities3 import *

torch.manual_seed(0)
np.random.seed(0)

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic

        self.p = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# configs
################################################################
TRAIN_PATH = 'data/piececonst_r421_N1024_smooth1.mat'
TEST_PATH = 'data/piececonst_r421_N1024_smooth2.mat'

parser = argparse.ArgumentParser()
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--ntest', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--modes', type=int, default=12)
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--r', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ntrain = args.ntrain
ntest = args.ntest
batch_size = args.batch_size
learning_rate = args.lr
weight_decay = args.weight_decay
epochs = args.epochs
modes = args.modes
width = args.width
r = args.r

iterations = epochs * (ntrain // batch_size)

h = int(((421 - 1) / r) + 1)
s = h

path = (
    f'darcy_fno_r421_N{ntrain}'
    f'_ep{epochs}'
    f'_m{modes}'
    f'_w{width}'
)

os.makedirs('model', exist_ok=True)
os.makedirs('pred', exist_ok=True)
os.makedirs('results', exist_ok=True)

path_model_best = f'model/{path}_best.pth'
path_model_last = f'model/{path}_last.pth'
path_pred = f'pred/{path}.mat'
path_metrics = f'results/{path}_metrics.txt'

################################################################
# load data and data normalization
################################################################
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

reader.load_file(TEST_PATH)
x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,s,s,1)
x_test = x_test.reshape(ntest,s,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
model = FNO2d(modes, modes, width).cuda()
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
best_test_l2 = float('inf')
best_epoch = -1

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(x.shape[0], s, s)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        loss = myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
        loss.backward()

        optimizer.step()
        scheduler.step()
        train_l2 += loss.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(x.shape[0], s, s)
            out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1)).item()

    train_l2/= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    if test_l2 < best_test_l2:
        best_test_l2 = test_l2
        best_epoch = ep
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'test_l2': test_l2,
            'modes': modes,
            'width': width,
            'ntrain': ntrain,
            'ntest': ntest,
            'r': r,
            's': s,
            'model_name': 'FNO2d',
        }, path_model_best)
        print(f'best checkpoint saved to {path_model_best}')

    print(ep, t2-t1, train_l2, test_l2)

torch.save({
    'epoch': epochs - 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'test_l2': test_l2,
    'modes': modes,
    'width': width,
    'ntrain': ntrain,
    'ntest': ntest,
    'r': r,
    's': s,
    'model_name': 'FNO2d',
}, path_model_last)

print(f'last checkpoint saved to {path_model_last}')
print(f'best epoch: {best_epoch}, best test l2: {best_test_l2}')

ckpt = torch.load(path_model_best, map_location='cuda')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'loaded best checkpoint from {path_model_best}')

pred = torch.zeros(ntest, s, s)
offset = 0

with torch.no_grad():
    for x, _ in test_loader:
        x = x.cuda()
        cur_bs = x.shape[0]

        out = model(x).reshape(cur_bs, s, s)
        out = y_normalizer.decode(out)

        pred[offset:offset + cur_bs] = out.cpu()
        offset += cur_bs

scipy.io.savemat(path_pred, {'pred': pred.numpy()})
print(f'prediction saved to {path_pred}')

with open(path_metrics, 'w', encoding='utf-8') as f:
    f.write('model_name: FNO2d\n')
    f.write(f'params: {count_params(model)}\n')
    f.write(f'modes: {modes}\n')
    f.write(f'width: {width}\n')
    f.write(f'ntrain: {ntrain}\n')
    f.write(f'ntest: {ntest}\n')
    f.write(f'r: {r}\n')
    f.write(f's: {s}\n')
    f.write(f'epochs: {epochs}\n')
    f.write(f'batch_size: {batch_size}\n')
    f.write(f'learning_rate: {learning_rate}\n')
    f.write(f'weight_decay: {weight_decay}\n')
    f.write(f'best_epoch: {best_epoch}\n')
    f.write(f'best_test_l2: {best_test_l2:.10e}\n')
    f.write(f'path_model_best: {path_model_best}\n')
    f.write(f'path_model_last: {path_model_last}\n')
    f.write(f'path_pred: {path_pred}\n')

print(f'metrics saved to {path_metrics}')

