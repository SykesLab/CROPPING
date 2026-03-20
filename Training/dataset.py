"""
PyTorch Dataset and DataLoader for Defocus Training — Scalar Head

Loads blurred images and scalar blur labels (σ in px) from metadata.csv.
No blur map PNGs or sharp images are loaded during training — only the
blurred image and the σ_px value from the metadata DataFrame.

DD-subnet datasets removed — scalar DME training only.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
import pandas as pd
import random


def _blur_column(df: pd.DataFrame) -> str:
    """Return the blur value column name: 'sigma_px' for direct mode, 'coc_px' for optical."""
    if 'sigma_px' in df.columns:
        return 'sigma_px'
    return 'coc_px'


def compute_max_blur_from_metadata(data_dir: Union[str, Path], margin_percent: float = 5.0) -> float:
    """
    Read metadata.csv and compute max blur (CoC or sigma) with safety margin.

    In direct mode (sigma_px column): no margin applied — the theoretical
    max_sigma is already the ceiling and random sampling can't exceed it.
    In optical mode (coc_px column): 5% margin applied as safety buffer.

    Args:
        data_dir: Directory containing metadata.csv
        margin_percent: Safety margin for optical mode (default 5%, ignored in direct mode)

    Returns:
        max blur value for normalisation
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / 'metadata.csv'

    if not metadata_path.exists():
        print(f"Warning: metadata.csv not found at {metadata_path}, using default max_blur=20.0")
        return 20.0

    try:
        df = pd.read_csv(metadata_path)
        col = _blur_column(df)
        if col not in df.columns:
            print(f"Warning: blur column not found in metadata.csv, using default max_blur=20.0")
            return 20.0

        actual_max = df[col].abs().max()
        is_direct = 'sigma_px' in df.columns
        blur_term = "sigma" if is_direct else "CoC"

        if is_direct:
            max_blur = actual_max
            print(f"Computed max {blur_term} from metadata: {actual_max:.2f} px (no margin - direct mode)")
        else:
            max_blur = actual_max * (1.0 + margin_percent / 100.0)
            print(f"Computed max {blur_term} from metadata: {actual_max:.2f} px")
            print(f"  With {margin_percent}% margin: {max_blur:.2f} px")

        return float(max_blur)
    except Exception as e:
        print(f"Warning: Error reading metadata.csv: {e}, using default max_blur=20.0")
        return 20.0


def compute_min_blur_from_metadata(data_dir: Union[str, Path]) -> float:
    """
    Read metadata.csv and compute min blur (CoC or sigma) from actual data.

    Args:
        data_dir: Directory containing metadata.csv

    Returns:
        min blur from data (at model scale)
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / 'metadata.csv'

    if not metadata_path.exists():
        print(f"Warning: metadata.csv not found at {metadata_path}, using default min_blur=0.0")
        return 0.0

    try:
        df = pd.read_csv(metadata_path)
        col = _blur_column(df)
        if col not in df.columns:
            print(f"Warning: blur column not found in metadata.csv, using default min_blur=0.0")
            return 0.0

        actual_min = df[col].abs().min()
        blur_term = "sigma" if 'sigma_px' in df.columns else "CoC"
        print(f"Computed min {blur_term} from metadata: {actual_min:.2f} px")

        return float(actual_min)
    except Exception as e:
        print(f"Warning: Error reading metadata.csv: {e}, using default min_blur=0.0")
        return 0.0


class DMEDataset(Dataset):
    """
    Dataset for scalar DME training.

    Loads only blurred images from disk + reads sigma_px from metadata.csv
    (already in memory). No blur map PNGs or sharp images are loaded.

    Returns (blur_img, blur_value_norm, blur_value_px) tuples.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        augment: bool = True,
        max_blur: float = 20.0
    ):
        """
        Args:
            data_dir: Directory containing blur/ subdir and metadata.csv
            augment: Whether to apply data augmentation
            max_blur: Maximum blur value for normalisation to [0, 1]
        """
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.max_blur = max_blur

        # Find blur images
        self.blur_dir = self.data_dir / 'blur'
        self.samples = sorted(list(self.blur_dir.glob('*.png')))

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {self.blur_dir}")

        # Load metadata (sigma_px / coc_px values)
        metadata_path = self.data_dir / 'metadata.csv'
        if not metadata_path.exists():
            raise ValueError(f"metadata.csv not found at {metadata_path}")

        self.metadata = pd.read_csv(metadata_path, index_col='index', dtype={'index': str})
        self.blur_col = _blur_column(self.metadata)
        self.is_direct_mode = 'sigma_px' in self.metadata.columns

        print(f"Loaded {len(self.samples)} samples from {data_dir}"
              f" ({'direct' if self.is_direct_mode else 'optical'} mode)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            (blur_img, blur_value_norm, blur_value_px)
            - blur_img: (1, H, W) tensor in [-1, 1]
            - blur_value_norm: (1,) tensor in [0, 1]
            - blur_value_px: (1,) tensor in pixels
        """
        blur_path = self.samples[idx]
        stem = blur_path.stem

        # Load only the blurred image
        blur = cv2.imread(str(blur_path), cv2.IMREAD_GRAYSCALE)
        if blur is None:
            raise ValueError(f"Failed to load {blur_path}")

        blur = blur.astype(np.float32) / 255.0

        # Read sigma_px from metadata (already in memory, no disk I/O)
        blur_value = self.metadata.loc[stem, self.blur_col]

        # Apply augmentation
        if self.augment:
            blur = self._augment(blur)

        # Normalise image to [-1, 1]
        blur = blur * 2.0 - 1.0

        # Convert to tensor
        blur = torch.from_numpy(blur).unsqueeze(0)  # (1, H, W)

        # Normalise blur value to [0, 1]
        blur_norm = torch.tensor([blur_value / self.max_blur], dtype=torch.float32)
        blur_px = torch.tensor([blur_value], dtype=torch.float32)

        return blur, blur_norm, blur_px

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
        if random.random() > 0.5:
            img = np.flipud(img).copy()
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k).copy()
        return img


def _create_stratified_split(data_dir: Path, max_blur: float, train_split: float, seed: int):
    """Create stratified train/val split based on blur bins."""
    import yaml

    metadata_path = data_dir / 'metadata.csv'
    df = pd.read_csv(metadata_path)
    col = _blur_column(df)
    blur_values = df[col].abs().values

    # Read config to check if min_blur filter was intentionally applied
    config_path = data_dir.parent / 'training_config.yaml'
    if not config_path.exists():
        config_path = data_dir.parent / 'optical_config.yaml'
    min_blur_intended = 0.0
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                data_section = config.get('data', {})
                min_blur_intended = data_section.get('min_blur_px', data_section.get('min_coc_px', 0.0))
            print(f"Loaded min_blur_px from config: {min_blur_intended}")
        else:
            print(f"Config not found at {config_path}, using default min_blur_px=0.0")
    except Exception as e:
        print(f"Error loading config: {e}")
        print(f"   Using default min_blur_px=0.0 (backward compatibility mode)")
        import traceback
        traceback.print_exc()

    min_blur_actual = blur_values.min()
    max_blur_actual = max_blur

    blur_term = "sigma" if 'sigma_px' in df.columns else "CoC"
    if min_blur_intended > 0.5:
        bin_size = (max_blur_actual - min_blur_actual) / 4.0
        bins = [(min_blur_actual + i * bin_size, min_blur_actual + (i + 1) * bin_size)
                for i in range(4)]
        print(f"{blur_term} bins (stratified split):")
        for i, (low, high) in enumerate(bins):
            print(f"  Bin {i+1}: [{low:.2f}, {high:.2f}] px")
    else:
        max_blur_ceil = int(np.ceil(max_blur))
        bin_size = max_blur_ceil / 4.0
        bins = [(i * bin_size, (i + 1) * bin_size) for i in range(4)]
        print(f"{blur_term} bins (stratified split): {bins}")

    # Assign each sample to a bin
    bin_assignments = np.zeros(len(blur_values), dtype=int)
    for i, (low, high) in enumerate(bins):
        mask = (blur_values >= low) & (blur_values < high)
        bin_assignments[mask] = i

    # For each bin, split into train/val
    train_indices = []
    val_indices = []

    np.random.seed(seed)
    for bin_idx in range(len(bins)):
        bin_mask = bin_assignments == bin_idx
        bin_sample_indices = np.where(bin_mask)[0]
        np.random.shuffle(bin_sample_indices)
        n_train = int(len(bin_sample_indices) * train_split)
        train_indices.extend(bin_sample_indices[:n_train].tolist())
        val_indices.extend(bin_sample_indices[n_train:].tolist())

    return train_indices, val_indices


def create_dme_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 50,
    num_workers: int = 4,
    train_split: float = 0.9,
    max_blur: float = 20.0,
    seed: int = 42,
    persistent_workers: bool = False,
    stratified: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for DME scalar training.

    Training loader uses augmentation. Validation loader does not.

    Args:
        stratified: If True, use stratified sampling to ensure each blur bin is
                   proportionally represented in both train and val sets.
    """
    data_dir = Path(data_dir)

    train_base = DMEDataset(data_dir=data_dir, augment=True, max_blur=max_blur)
    val_base = DMEDataset(data_dir=data_dir, augment=False, max_blur=max_blur)

    if stratified:
        train_idx, val_idx = _create_stratified_split(data_dir, max_blur, train_split, seed)
    else:
        total = len(train_base)
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(total, generator=generator)
        train_size = int(total * train_split)
        train_idx = perm[:train_size].tolist()
        val_idx = perm[train_size:].tolist()

    train_dataset = torch.utils.data.Subset(train_base, train_idx)
    val_dataset = torch.utils.data.Subset(val_base, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )

    return train_loader, val_loader


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic"

    print("Testing DMEDataset (scalar head)...")

    try:
        dataset = DMEDataset(data_dir, max_blur=13.72)

        blur, norm, px = dataset[0]
        blur_term = "sigma" if dataset.is_direct_mode else "CoC"
        print(f"Blur shape: {blur.shape}")
        print(f"{blur_term} normalised: {norm.item():.4f}")
        print(f"{blur_term} raw: {px.item():.2f} px")

        print("\nTesting DataLoader...")
        train_loader, val_loader = create_dme_dataloaders(data_dir, batch_size=4, max_blur=13.72)

        batch = next(iter(train_loader))
        print(f"Batch blur shape: {batch[0].shape}")
        print(f"Batch norm shape: {batch[1].shape}")
        print(f"Batch px shape: {batch[2].shape}")

        print("\nAll tests passed!")

    except Exception as e:
        print(f"Error: {e}")
        print("Generate synthetic data first with: python synthetic_blur.py")
