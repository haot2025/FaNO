
import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_pair(x, default=(4, 4)):
    if x is None:
        return default
    if isinstance(x, str):
        x = x.strip().replace("'", "").replace('"', "")
        parts = x.split(",")
        if len(parts) == 1:
            v = int(parts[0])
            return (v, v)
        return (int(parts[0]), int(parts[1]))
    if isinstance(x, (tuple, list)):
        if len(x) == 1:
            return (int(x[0]), int(x[0]))
        return (int(x[0]), int(x[1]))
    v = int(x)
    return (v, v)


class SpectralMix2d(nn.Module):
    def __init__(self, channels, modes1=16, modes2=16):
        super().__init__()
        self.channels = int(channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)

        scale = 1.0 / (channels * channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(channels, channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(channels, channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            b, c, h, w // 2 + 1,
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


class LSMBlock(nn.Module):
    def __init__(self, d_model, heads, modes1=16, modes2=16, mlp_ratio=2):
        super().__init__()

        self.norm1 = nn.GroupNorm(1, d_model)
        self.spectral = SpectralMix2d(d_model, modes1=modes1, modes2=modes2)
        self.local = nn.Conv2d(d_model, d_model, kernel_size=1)

        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            batch_first=True,
        )

        self.norm3 = nn.GroupNorm(1, d_model)
        self.mlp = nn.Sequential(
            nn.Conv2d(d_model, d_model * mlp_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model * mlp_ratio, d_model, kernel_size=1),
        )

    def forward(self, x, latent_tokens):
        # spectral/local grid mixing
        x = x + self.spectral(self.norm1(x)) + self.local(x)

        # latent token global mixing
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()  # B HW C
        tokens_norm = self.norm2(tokens)
        latent = latent_tokens.unsqueeze(0).expand(b, -1, -1).contiguous()

        attn_out, _ = self.attn(tokens_norm, latent, latent, need_weights=False)
        tokens = tokens + attn_out
        x = tokens.transpose(1, 2).contiguous().view(b, c, h, w)

        # channel MLP
        x = x + self.mlp(self.norm3(x))
        return x


class lsm(nn.Module):
    """
    Lightweight Latent Spectral Model baseline for 2D operator learning.

    It combines:
      1. patch embedding,
      2. spectral mixing on the latent grid,
      3. learned global latent tokens,
      4. transposed-conv decoding back to dense fields.
    """

    def __init__(self, params):
        super().__init__()

        self.in_channels = int(params.in_dim)
        self.out_channels = int(params.out_dim)

        self.d_model = int(getattr(params, "d_model", getattr(params, "embed_cut", 84)))
        self.num_basis = int(getattr(params, "num_basis", 12))
        self.num_token = int(getattr(params, "num_token", 4))
        self.heads = int(getattr(params, "lsm_head", 7))

        if self.d_model % self.heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by lsm_head={self.heads}")

        self.patch_size = _parse_pair(getattr(params, "patch_size", "4,4"), default=(4, 4))
        self.padding_pair = _parse_pair(getattr(params, "padding", "0,0"), default=(0, 0))

        self.depth = int(getattr(params, "lsm_depth", 4))
        self.mlp_ratio = int(getattr(params, "lsm_mlp_ratio", 2))

        mode_cut = int(getattr(params, "mode_cut", 32))
        self.modes1 = int(getattr(params, "lsm_modes1", min(16, mode_cut)))
        self.modes2 = int(getattr(params, "lsm_modes2", min(16, mode_cut)))

        patch_in_channels = self.in_channels + 2

        self.patch_embed = nn.Conv2d(
            patch_in_channels,
            self.d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.latent_tokens = nn.Parameter(torch.randn(self.num_token, self.d_model) * 0.02)

        self.blocks = nn.ModuleList([
            LSMBlock(
                d_model=self.d_model,
                heads=self.heads,
                modes1=self.modes1,
                modes2=self.modes2,
                mlp_ratio=self.mlp_ratio,
            )
            for _ in range(self.depth)
        ])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.d_model,
                self.d_model,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            nn.GELU(),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.d_model, self.out_channels, kernel_size=1),
        )

    def get_grid(self, x):
        b, _, h, w = x.shape
        device = x.device
        dtype = x.dtype

        gridx = torch.linspace(0, 1, h, device=device, dtype=dtype)
        gridx = gridx.view(1, 1, h, 1).repeat(b, 1, 1, w)

        gridy = torch.linspace(0, 1, w, device=device, dtype=dtype)
        gridy = gridy.view(1, 1, 1, w).repeat(b, 1, h, 1)

        return torch.cat([gridx, gridy], dim=1)

    def pad_to_patch(self, x):
        b, c, h, w = x.shape
        ph, pw = self.patch_size

        pad_h = (ph - h % ph) % ph
        pad_w = (pw - w % pw) % pw

        extra_h, extra_w = self.padding_pair
        pad_h += extra_h
        pad_w += extra_w

        if pad_h or pad_w:
            x = F.pad(x, [0, pad_w, 0, pad_h])

        return x, h, w

    def forward(self, x):
        nhwc = False
        if x.dim() == 4 and x.shape[1] not in (1, 2, 3, 4, 5) and x.shape[-1] in (1, 2, 3, 4, 5):
            nhwc = True
            x = x.permute(0, 3, 1, 2).contiguous()

        x, h0, w0 = self.pad_to_patch(x)

        grid = self.get_grid(x)
        x = torch.cat([x, grid], dim=1)

        z = self.patch_embed(x)

        for block in self.blocks:
            z = block(z, self.latent_tokens)

        out = self.decoder(z)
        out = out[..., :h0, :w0].contiguous()

        if nhwc:
            out = out.permute(0, 2, 3, 1).contiguous()

        return out
