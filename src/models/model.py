import torch
from torch import nn

from .edge_freq import EdgeFrequencyAugment
from .film import FiLM
from .modules import ModalitySE
from .unet3d import UNet3D


class EFFUNet(nn.Module):
    def __init__(
        self,
        in_modalities=4,
        base_channels=32,
        edge_enabled=True,
        freq_enabled=True,
        modality_attention=True,
        film_metadata=True,
        meta_dim=2,
        out_channels=3,
    ):
        super().__init__()
        self.modality_attention = ModalitySE(in_modalities) if modality_attention else None
        self.augment = EdgeFrequencyAugment(edge_enabled, freq_enabled)

        extra = 0
        if edge_enabled:
            extra += in_modalities
        if freq_enabled:
            extra += in_modalities
        in_channels = in_modalities + extra

        film = FiLM(meta_dim, base_channels * 16) if film_metadata else None
        self.unet = UNet3D(in_channels, out_channels, base_channels, film=film)

    def forward(self, x, meta=None):
        if self.modality_attention is not None:
            x = self.modality_attention(x)
        x = self.augment(x)
        return self.unet(x, meta=meta)


def brats_label_map(y):
    # BraTS labels: 0=background, 1=NT, 2=ED, 4=ET
    wt = (y > 0).float()
    tc = ((y == 1) | (y == 4)).float()
    et = (y == 4).float()
    return torch.stack([wt, tc, et], dim=1)
