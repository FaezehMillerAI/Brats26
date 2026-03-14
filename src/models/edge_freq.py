import torch
from torch import nn


def _sobel_kernel_3d(device, dtype):
    k = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    kx = k.view(1, 1, 1, 3, 3)
    ky = k.view(1, 1, 3, 1, 3)
    kz = k.view(1, 1, 3, 3, 1)
    return kx, ky, kz


class EdgeFrequencyAugment(nn.Module):
    def __init__(self, enable_edge: bool, enable_freq: bool):
        super().__init__()
        self.enable_edge = enable_edge
        self.enable_freq = enable_freq

    def forward(self, x):
        feats = [x]

        if self.enable_edge:
            device = x.device
            dtype = x.dtype
            kx, ky, kz = _sobel_kernel_3d(device, dtype)
            grads = []
            for c in range(x.shape[1]):
                xc = x[:, c : c + 1]
                gx = nn.functional.conv3d(xc, kx, padding=(0, 1, 1))
                gy = nn.functional.conv3d(xc, ky, padding=(1, 0, 1))
                gz = nn.functional.conv3d(xc, kz, padding=(1, 1, 0))
                g = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + 1e-6)
                grads.append(g)
            feats.append(torch.cat(grads, dim=1))

        if self.enable_freq:
            fft = torch.fft.fftn(x, dim=(-3, -2, -1))
            mag = torch.log1p(torch.abs(fft))
            feats.append(mag.real)

        return torch.cat(feats, dim=1)
