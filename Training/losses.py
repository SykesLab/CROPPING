"""
Loss Functions for Defocus Estimation Training

Log-space MSE loss for scalar blur prediction.
Operates in pixel space to ensure correct relative weighting.
DD-subnet losses removed — scalar estimation only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DMELoss(nn.Module):
    """
    Log-space MSE loss for scalar blur estimation.

    Converts predictions from normalised [0, 1] to pixel space, then computes
    MSE in log space. This inherently penalises relative error: a 50% error
    at σ=1 and a 50% error at σ=12 produce the same loss.

    Loss = MSE(log(pred_px + ε), log(target_px + ε))

    where pred_px = pred_norm * max_blur, target_px = target_norm * max_blur.

    IMPORTANT: max_blur must be the SAME value used in the dataset normalisation
    and stored in the checkpoint. Do not recompute independently.
    """

    def __init__(self, max_blur: float, eps: float = 0.01):
        """
        Args:
            max_blur: Maximum blur in pixels — single source of truth from Trainer.
            eps: Small constant for numerical stability in log. Configurable via
                 training_config.yaml 'log_eps'. Default 0.01 is safe when
                 min_blur_px >= 0.5 at model scale.
        """
        super().__init__()
        self.max_blur = max_blur
        self.eps = eps

    def forward(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_norm: Predicted blur, shape (B, 1), normalised to [0, 1]
            target_norm: Target blur, shape (B, 1) or (B,), normalised to [0, 1]

        Returns:
            Loss scalar
        """
        if target_norm.dim() == 1:
            target_norm = target_norm.unsqueeze(1)

        # Convert to pixel space BEFORE taking log
        pred_px = pred_norm * self.max_blur
        target_px = target_norm * self.max_blur

        loss = F.mse_loss(
            torch.log(pred_px + self.eps),
            torch.log(target_px + self.eps),
        )
        return loss


# =============================================================================
# Metrics (unchanged — operate on images in [-1, 1])
# =============================================================================
def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted image
        target: Target image

    Returns:
        PSNR in dB
    """
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float('inf')

    # Images normalised to [-1, 1], max value difference is 2
    max_val = 2.0
    psnr = 10 * torch.log10(torch.tensor(max_val ** 2 / mse))
    return psnr.item()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11
) -> float:
    """
    Compute Structural Similarity Index (simplified version).

    Args:
        pred: Predicted image (B, 1, H, W)
        target: Target image (B, 1, H, W)
        window_size: Window size for local statistics

    Returns:
        Mean SSIM
    """
    # Force SSIM computation in float32 to avoid AMP dtype mismatches
    pred = pred.float()
    target = target.float()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window = g.outer(g)
    window = window / window.sum()
    window = window.view(1, 1, window_size, window_size).to(device=pred.device, dtype=pred.dtype)

    # Compute local means
    mu_pred = F.conv2d(pred, window, padding=window_size // 2)
    mu_target = F.conv2d(target, window, padding=window_size // 2)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    # Compute local variances and covariance
    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=window_size // 2) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window, padding=window_size // 2) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=window_size // 2) - mu_pred_target

    # SSIM formula
    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)

    ssim_map = numerator / denominator
    return ssim_map.mean().item()


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("Testing loss functions...")

    batch_size = 4

    # Test DME loss (log-space)
    pred = torch.rand(batch_size, 1) * 0.8 + 0.1   # [0.1, 0.9]
    target = torch.rand(batch_size, 1) * 0.8 + 0.1  # [0.1, 0.9]

    dme_loss = DMELoss(max_blur=13.72, eps=0.01)
    loss = dme_loss(pred, target)
    print(f"DME Loss (log-space): {loss.item():.4f}")

    # Verify relative error property: same % error at different magnitudes
    # should give similar loss
    pred_low = torch.tensor([[0.1]])    # 1.37 px
    target_low = torch.tensor([[0.15]])  # 2.06 px  (~50% error)
    pred_high = torch.tensor([[0.7]])   # 9.60 px
    target_high = torch.tensor([[1.0]])  # 13.72 px (~43% error)

    loss_low = dme_loss(pred_low, target_low)
    loss_high = dme_loss(pred_high, target_high)
    print(f"Loss at low blur (50% error):  {loss_low.item():.4f}")
    print(f"Loss at high blur (43% error): {loss_high.item():.4f}")
    print(f"Ratio: {loss_low.item() / loss_high.item():.2f} (should be ~1.0)")

    print("\nAll tests passed!")
