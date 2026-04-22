"""
Defocus Estimation Neural Network — Scalar Head Architecture

Predicts a single scalar blur value (σ) from a blurred input image.
Conv backbone extracts features, 4×4 adaptive pooling preserves spatial
gradient signal, and a single FC layer maps to Sigmoid output in [0, 1].

Based on Wang et al. (2022), Physics of Fluids, 34(7), 073301.
DD-subnet (deblurring) removed — scalar blur estimation only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class ResBlock(nn.Module):
    """Residual block: Conv -> LReLU -> Conv + skip connection."""

    def __init__(self, channels: int, kernel_size: int = 3, alpha: float = 0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.lrelu = nn.LeakyReLU(negative_slope=alpha, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class DMESubnet(nn.Module):
    """
    Defocus Map Estimation Subnet — Scalar Head.

    Input: Blurred image (B, 1, H, W)
    Output: Scalar blur prediction (B, 1), normalised to [0, 1]

    Architecture:
        Conv7x7 → LeakyReLU → ResBlocks → AdaptiveAvgPool(4,4) → Flatten → FC → Sigmoid
    """

    def __init__(
        self,
        in_channels: int = 1,
        initial_filters: int = 32,
        num_res_blocks: int = 2,
        kernel_size: int = 3,
        alpha: float = 0.2,
        pool_size: int = 4
    ):
        super().__init__()

        self.initial_filters = initial_filters

        # Convolutional backbone (same as original)
        self.conv_in = nn.Conv2d(in_channels, initial_filters, 7, padding=3)
        self.lrelu = nn.LeakyReLU(negative_slope=alpha, inplace=False)

        self.res_blocks = nn.Sequential(*[
            ResBlock(initial_filters, kernel_size, alpha)
            for _ in range(num_res_blocks)
        ])

        # Scalar head: 4×4 adaptive pooling → flatten → FC → Sigmoid
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.fc = nn.Linear(initial_filters * pool_size * pool_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lrelu(self.conv_in(x))
        out = self.res_blocks(out)
        out = self.pool(out)                    # (B, C, 4, 4)
        out = out.flatten(1)                    # (B, C*4*4)
        out = self.sigmoid(self.fc(out))        # (B, 1)
        return out


class DefocusNet(nn.Module):
    """Scalar defocus estimation network (DME-subnet only)."""

    def __init__(self, dme_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        dme_config = dme_config or {}
        self.dme_subnet = DMESubnet(**dme_config)

    def forward(self, blur: torch.Tensor) -> torch.Tensor:
        return self.dme_subnet(blur)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DefocusNet':
        net_config = config.get('network', {})

        dme_config = {
            'initial_filters': net_config.get('dme', {}).get('initial_filters', 32),
            'num_res_blocks': net_config.get('dme', {}).get('num_res_blocks', 2),
            'kernel_size': net_config.get('dme', {}).get('kernel_size', 3),
        }

        return cls(dme_config=dme_config)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 1, 256, 256)):
    print("="*60)
    print(f"Model: {model.__class__.__name__}")
    print("="*60)

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Approximate size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    if hasattr(model, 'dme_subnet'):
        print(f"  DME-subnet: {count_parameters(model.dme_subnet):,} params")

    print("="*60)


if __name__ == "__main__":
    print("Testing DefocusNet (scalar head)...")

    model = DefocusNet()
    model_summary(model)

    batch_size = 4
    img_size = 256
    dummy_input = torch.randn(batch_size, 1, img_size, img_size)

    print(f"\nInput shape: {dummy_input.shape}")

    scalar_pred = model(dummy_input)
    print(f"Scalar prediction shape: {scalar_pred.shape}")
    print(f"Scalar prediction range: [{scalar_pred.min().item():.4f}, {scalar_pred.max().item():.4f}]")

    print("\nAll tests passed!")
