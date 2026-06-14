import torch
import torch.nn as nn
from .basics import _get_act


class conv_fano(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, persistent_ratio=0.25):
        super(conv_fano, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.persistent_ratio = persistent_ratio

        self.scale = 1.0 / (in_channels * out_channels)

        # persistent/global channels
        self.persistent_channels = max(1, int(round(out_channels * persistent_ratio)))
        # dynamic/local channels
        self.dynamic_channels = out_channels - self.persistent_channels

        if self.dynamic_channels <= 0:
            raise ValueError(
                f"persistent_ratio={persistent_ratio} too large for out_channels={out_channels}"
            )

        # dynamic response
        self.weights1 = nn.Parameter(
            torch.randn(in_channels, self.dynamic_channels, modes1, modes2, dtype=torch.complex64) * self.scale
        )
        self.weights2 = nn.Parameter(
            torch.randn(in_channels, self.dynamic_channels, modes1, modes2, dtype=torch.complex64) * self.scale
        )

        # persistent response
        self.weights_x = nn.Parameter(
            torch.randn(in_channels, self.persistent_channels, dtype=torch.float32) * self.scale
        )
        self.weights3 = nn.Parameter(
            torch.randn(1, self.persistent_channels, modes1, modes2, dtype=torch.complex64) * self.scale
        )
        self.weights4 = nn.Parameter(
            torch.randn(1, self.persistent_channels, modes1, modes2, dtype=torch.complex64) * self.scale
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # dynamic branch
        weight = torch.zeros(
            self.in_channels, self.dynamic_channels, H, W // 2 + 1,
            dtype=x_ft.dtype, device=x.device
        )
        weight[:, :, :self.modes1, :self.modes2] = self.weights1
        weight[:, :, -self.modes1:, :self.modes2] = self.weights2

        out_ft_dynamic = torch.einsum('bchw,cohw->bohw', x_ft, weight)

        # persistent branch
        weight0 = torch.zeros(
            1, self.persistent_channels, H, W // 2 + 1,
            dtype=x_ft.dtype, device=x.device
        )
        weight0[:, :, :self.modes1, :self.modes2] = self.weights3
        weight0[:, :, -self.modes1:, :self.modes2] = self.weights4

        pooled = torch.mean(x, dim=[2, 3], keepdim=True)   # [B, C, 1, 1]
        persistent_coef = torch.einsum('io,bixy->boxy', self.weights_x, pooled)
        out_ft_persistent = weight0 * persistent_coef

        out_ft = torch.cat([out_ft_dynamic, out_ft_persistent], dim=1)
        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        return x


class FaNO_Net(nn.Module):
    def __init__(self, modes1, modes2,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=1,
                 dropout=0,
                 activation='tanh',
                 mean_constraint=False,
                 persistent_ratio=0.25):
        super(FaNO_Net, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.persistent_ratio = persistent_ratio

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers

        self.fc0 = nn.Linear(in_dim, self.layers[0])

        self.sp_convs = nn.ModuleList([
            conv_fano(
                in_size, out_size, mode1_num, mode2_num,
                persistent_ratio=self.persistent_ratio
            )
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)
        ])

        self.dropout = nn.Dropout(p=dropout)

        self.ws = nn.ModuleList([
            nn.Conv1d(in_size, out_size, 1)
            for in_size, out_size in zip(self.layers, self.layers[1:])
        ])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.activation = _get_act(activation)
        self.mean_constraint = mean_constraint

    def forward(self, x):
        '''
        input  x: (b,c,h,w)
        output x: (b,1,h,w)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        # repo input is [B, C, H, W]
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(
                batchsize, self.layers[i + 1], size_x, size_y
            )
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
            x = self.dropout(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x.permute(0, 3, 1, 2)

        if self.mean_constraint:
            x = x - torch.mean(x, dim=(-2, -1), keepdim=True)

        return x


def FaNO_model(params):
    if params.mode_cut > 0:
        params.modes1 = [params.mode_cut] * len(params.modes1)
        params.modes2 = [params.mode_cut] * len(params.modes2)

    if params.embed_cut > 0:
        params.layers = [params.embed_cut] * len(params.layers)

    if params.fc_cut > 0 and params.embed_cut > 0:
        params.fc_dim = params.embed_cut * params.fc_cut

    input_dim = params.in_dim

    persistent_ratio = 0.25
    if hasattr(params, "persistent_ratio"):
        persistent_ratio = float(params.persistent_ratio)

    return FaNO_Net(
        params.modes1, params.modes2,
        layers=params.layers, fc_dim=params.fc_dim,
        in_dim=input_dim, out_dim=params.out_dim,
        dropout=params.dropout,
        activation='gelu',
        mean_constraint=(params.loss_func == 'pde'),
        persistent_ratio=persistent_ratio
    )


def fano(params):
    """Backward-compatible factory entry point."""
    return FaNO_model(params)
