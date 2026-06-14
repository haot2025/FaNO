
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            b, self.out_channels, h, w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weights1[:, :, :m1, :m2],
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2],
            self.weights2[:, :, :m1, :m2],
        )

        return torch.fft.irfft2(out_ft, s=(h, w))


class FNOTrunk2d(nn.Module):
    def __init__(self, in_channels, width, modes1, modes2, depth=4, padding=0):
        super().__init__()
        self.padding = int(padding)

        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)

        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2)
            for _ in range(depth)
        ])
        self.pointwise_layers = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1)
            for _ in range(depth)
        ])

    def forward(self, x):
        x = self.lift(x)

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for i, (spec, pw) in enumerate(zip(self.spectral_layers, self.pointwise_layers)):
            x = spec(x) + pw(x)
            if i != len(self.spectral_layers) - 1:
                x = F.gelu(x)

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]

        return x


class msfno(nn.Module):
    """
    Full MscaleFNO-style baseline:
    parallel full FNO trunks with scale-specific coordinate grids,
    then concatenate branch features and fuse.
    """

    def __init__(self, params):
        super().__init__()

        self.in_channels = int(params.in_dim)
        self.out_channels = int(params.out_dim)

        self.width = int(getattr(params, "msfno_width", getattr(params, "embed_cut", 96)))
        self.fc_dim = int(getattr(params, "fc_dim", 128))

        self.depth = int(getattr(params, "msfno_depth", 4))
        self.padding = int(getattr(params, "msfno_padding", 0))

        mode_cut = int(getattr(params, "mode_cut", 32))
        self.modes1 = int(getattr(params, "msfno_modes1", mode_cut))
        self.modes2 = int(getattr(params, "msfno_modes2", mode_cut))

        scales_raw = getattr(params, "msfno_scales", "1,2,4")
        if isinstance(scales_raw, str):
            self.scales = [float(x) for x in scales_raw.split(",")]
        else:
            self.scales = [float(x) for x in scales_raw]

        self.use_grid = str(getattr(params, "msfno_use_grid", "True")).lower() in ("true", "1", "yes")

        branch_in_channels = self.in_channels + 2 if self.use_grid else self.in_channels

        self.branches = nn.ModuleList([
            FNOTrunk2d(
                in_channels=branch_in_channels,
                width=self.width,
                modes1=self.modes1,
                modes2=self.modes2,
                depth=self.depth,
                padding=self.padding,
            )
            for _ in self.scales
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(self.width * len(self.scales), self.fc_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.fc_dim, self.out_channels, kernel_size=1),
        )

    def get_grid(self, x, scale):
        b, _, h, w = x.shape
        device = x.device
        dtype = x.dtype

        gridx = torch.linspace(0, 1, h, device=device, dtype=dtype)
        gridx = gridx.view(1, 1, h, 1).repeat(b, 1, 1, w)

        gridy = torch.linspace(0, 1, w, device=device, dtype=dtype)
        gridy = gridy.view(1, 1, 1, w).repeat(b, 1, h, 1)

        return torch.cat([gridx * scale, gridy * scale], dim=1)

    def forward(self, x):
        nhwc = False
        if x.dim() == 4 and x.shape[1] not in (1, 2, 3, 4, 5) and x.shape[-1] in (1, 2, 3, 4, 5):
            nhwc = True
            x = x.permute(0, 3, 1, 2).contiguous()

        feats = []
        for scale, branch in zip(self.scales, self.branches):
            if self.use_grid:
                grid = self.get_grid(x, scale)
                xin = torch.cat([x, grid], dim=1)
            else:
                xin = x
            feats.append(branch(xin))

        out = self.fuse(torch.cat(feats, dim=1))

        if nhwc:
            out = out.permute(0, 2, 3, 1).contiguous()

        return out
