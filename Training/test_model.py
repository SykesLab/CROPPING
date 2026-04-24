"""
Model Testing Script for Defocus Estimation (DME-only)

Tests blur estimation across N samples with metrics.

Usage:
    python test_model.py --model checkpoints/dme_best.pth --data data/synthetic --samples 100
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from model import DefocusNet
from physics import defocus_uncertainty
from synthetic_blur import BlurCalculator, BlurParams

logger = logging.getLogger(__name__)


class TestDataset(Dataset):
    """Dataset for batched testing."""

    def __init__(self, sample_paths: List[Path], data_dir: Path, max_blur: float):
        self.sample_paths = sample_paths
        self.data_dir = data_dir
        self.max_blur = max_blur

        # New datasets use 'blur_map/', fall back to 'coc_map/' for older data
        bm = data_dir / 'blur_map'
        self.blur_map_dir = bm if bm.exists() else data_dir / 'coc_map'

        self.metadata = None
        metadata_path = data_dir / 'metadata.csv'
        if metadata_path.exists():
            self.metadata = pd.read_csv(metadata_path, index_col='index', dtype={'index': str})

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]
        stem = sample_path.stem

        blur_path = sample_path
        blur_map_path = self.blur_map_dir / f'{stem}.png'
        sharp_path = self.data_dir / 'sharp' / f'{stem}.png'

        blur = cv2.imread(str(blur_path), cv2.IMREAD_GRAYSCALE)
        blur_map = cv2.imread(str(blur_map_path), cv2.IMREAD_GRAYSCALE)
        sharp = cv2.imread(str(sharp_path), cv2.IMREAD_GRAYSCALE)

        if blur is None or blur_map is None or sharp is None:
            raise ValueError(f"Failed to load sample {stem}")

        blur = blur.astype(np.float32) / 255.0
        blur_map_img = blur_map.astype(np.float32) / 255.0
        sharp = sharp.astype(np.float32) / 255.0

        if self.metadata is not None:
            blur_col = 'sigma_px' if 'sigma_px' in self.metadata.columns else 'coc_px'
            blur_value_gt = self.metadata.loc[stem, blur_col]
        else:
            blur_value_gt = blur_map_img.mean() * self.max_blur

        # Normalize images to [-1, 1], blur maps stay [0, 1]
        blur = blur * 2.0 - 1.0
        sharp = sharp * 2.0 - 1.0

        # Convert to tensors (no batch dimension yet, DataLoader will add it)
        blur = torch.from_numpy(blur).unsqueeze(0)  # [1, H, W]
        blur_map = torch.from_numpy(blur_map_img).unsqueeze(0)  # [1, H, W]
        sharp = torch.from_numpy(sharp).unsqueeze(0)  # [1, H, W]

        return {
            'blur': blur,
            'blur_map': blur_map,
            'sharp': sharp,
            'blur_value': blur_value_gt,
            'sample_name': stem
        }


class ModelTester:
    """Test trained defocus estimation models."""

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = 'auto'
    ):
        self.model_path = Path(model_path)

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        info = [f"Loaded {Path(model_path).name}",
                f"epoch={checkpoint.get('epoch', '?')}"]
        if 'val_mae_px' in checkpoint:
            info.append(f"val_mae={checkpoint['val_mae_px']:.4f}px")
        if 'val_psnr' in checkpoint:
            info.append(f"val_psnr={checkpoint['val_psnr']:.2f}dB")
        if 'val_ssim' in checkpoint:
            info.append(f"val_ssim={checkpoint['val_ssim']:.4f}")
        logger.info(" | ".join(info))

        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            raise ValueError("Config not found. Provide config_path.")

        # Detect training mode and set terminology
        self.training_mode = self.config.get('training', {}).get('training_mode', 'optical')
        self.blur_term = "σ" if self.training_mode == "direct" else "CoC"
        self.blur_col = "sigma" if self.training_mode == "direct" else "coc"

        self.model = DefocusNet.from_config(self.config).to(self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model weights")
        elif 'dme_state_dict' in checkpoint:
            self.model.dme_subnet.load_state_dict(checkpoint['dme_state_dict'])
            print("Loaded DME-subnet weights")

        self.model.eval()

        # Optical parameters
        self.optical_params = BlurParams.from_config(self.config)
        self.blur_calc = BlurCalculator(self.optical_params)

        # Get max blur for denormalization (prefer checkpoint value for consistency)
        ckpt_max = checkpoint.get('max_blur', checkpoint.get('max_coc'))
        if ckpt_max is not None:
            self.max_blur = ckpt_max
            print(f"ℹ Using max_blur from checkpoint: {self.max_blur:.2f} px")
        else:
            data_cfg = self.config.get('data', {})
            self.max_blur = data_cfg.get('coc_range_px', [1, 20])[1]
            print(
                f"⚠ max_blur not found in checkpoint, using config default: {self.max_blur:.2f} px")

        print(f"Device: {self.device}")

        # LOO-CV calibration uncertainties (loaded from config if available)
        training_cfg = self.config.get('training', {})
        self.rho_std = float(training_cfg.get('rho_std', 0.0))
        self.sigma_0_std = float(training_cfg.get('sigma_0_std', 0.0))
        # Also try direct calibration params for defocus uncertainty
        self.rho_direct = float(training_cfg.get('rho_direct', 0.0))
        self.sigma_0_direct = float(training_cfg.get('sigma_0', 0.0))
        if self.rho_std > 0:
            print(
                f"Calibration uncertainty: rho_std={self.rho_std:.4f}, sigma_0_std={self.sigma_0_std:.4f}")

        # Calculate bin weights from beta distribution
        self.bin_weights = self._calculate_bin_weights_from_beta()

    def _calculate_bin_weights_from_beta(self) -> list:
        from utils import calculate_bin_weights_from_beta
        return calculate_bin_weights_from_beta(self.config)

    def denormalize_blur(self, blur_normalized) -> float:
        """Convert normalized blur [0, 1] to pixels. Delegates to physics module."""
        val = blur_normalized.item() if hasattr(blur_normalized, 'item') else float(blur_normalized)
        from physics import denormalise_label
        return denormalise_label(val, self.max_blur)

    def _get_bins(self):
        max_blur_ceil = int(np.ceil(self.max_blur))
        bin_size = max_blur_ceil / 4.0
        return [(i * bin_size, (i + 1) * bin_size) for i in range(4)]

    def test_dme_only(
        self,
        data_dir: str,
        num_samples: int = 100,
        output_dir: Optional[str] = None,
        num_visual_samples: int = 16,
        viz_percent: float = 10.0,
        batch_size: int = 1,
        num_workers: int = 0,
        num_worst_px: int = 0,
        num_worst_pct: int = 0,
        num_worst: int = 0,  # Legacy parameter for backward compatibility
        min_blur_filter: Optional[float] = None,
        filter_worst_pct: bool = False,
        filter_metrics: bool = False,
        exclude_from_test: bool = False,
        filter_intervals: bool = False,
        filter_plots: bool = False,
        create_subfolder: bool = True,
    ) -> pd.DataFrame:
        """
        Test DME-subnet only across N samples.

        Returns DataFrame with per-sample results and computes aggregate metrics.

        Args:
            data_dir: Path to test data
            num_samples: Number of samples to test
            output_dir: Directory to save results and visualizations
            num_visual_samples: Number of samples to create visual comparisons for
            viz_percent: Percentage of samples to visualize (1-100)
            batch_size: Batch size for inference (default: 1)
            num_workers: Number of worker threads for data loading (default: 0)
            num_worst_px: Number of worst-case samples by blur error px to save (0 to disable)
            num_worst_pct: Number of worst-case samples by blur error % to save (0 to disable)
            num_worst: Legacy parameter (use num_worst_px/pct instead)
            min_blur_filter: Minimum |blur| threshold for filtering. When set, bins/intervals start from this value instead of 0
            filter_worst_pct: Apply min_blur_filter to worst % error cases (exclude low blur samples)
            filter_metrics: Show additional filtered metrics comparison (exclude low blur samples)
            filter_intervals: DEPRECATED - binning and intervals are always shown
            filter_plots: DEPRECATED - plots are always generated
        """
        # Handle legacy parameter
        if num_worst > 0 and num_worst_px == 0 and num_worst_pct == 0:
            num_worst_px = num_worst
            num_worst_pct = num_worst
        print("\n" + "=" * 60)
        print("DME-Only Testing Mode")
        print("=" * 60)

        data_dir = Path(data_dir)
        print(f"Test data directory: {data_dir}")
        print(f"Using max_blur from checkpoint: {self.max_blur:.2f} px")

        blur_dir = data_dir / 'blur'

        all_samples = sorted(list(blur_dir.glob('*.png')))
        total_available = len(all_samples)

        # Show a sample of file names to verify we're loading the right data
        if len(all_samples) > 0:
            print(
                f"Sample files: {all_samples[0].name}, {all_samples[min(1, len(all_samples)-1)].name}, ...")

        if num_samples > 0 and num_samples < total_available:
            all_samples = all_samples[:num_samples]

        # Apply blur exclusion filter if enabled
        if exclude_from_test and min_blur_filter is not None:
            # Load metadata CSV to get blur values
            metadata_csv = data_dir / 'metadata.csv'

            if metadata_csv.exists():
                metadata_df = pd.read_csv(metadata_csv)

                # Create a mapping from sample index to blur value
                # Normalize index to match filename format (with leading zeros removed for matching)
                blur_col_name = 'sigma_px' if 'sigma_px' in metadata_df.columns else 'coc_px'
                blur_lookup = {}
                for _, row in metadata_df.iterrows():
                    # Convert to int then str to remove leading zeros
                    sample_idx = str(int(row['index']))
                    blur_px = float(row[blur_col_name])
                    blur_lookup[sample_idx] = blur_px

                # Filter samples based on blur values from metadata
                filtered_samples = []
                excluded_count = 0
                missing_metadata_count = 0

                for sample_path in all_samples:
                    # Extract sample index from filename (e.g., "000123.png" -> "123")
                    sample_idx = str(int(sample_path.stem))  # Remove leading zeros for matching

                    if sample_idx in blur_lookup:
                        blur_value = blur_lookup[sample_idx]

                        # Include sample only if |blur| >= min_blur_filter
                        if abs(blur_value) >= min_blur_filter:
                            filtered_samples.append(sample_path)
                        else:
                            excluded_count += 1
                    else:
                        # If no metadata found, include the sample (avoid data loss)
                        missing_metadata_count += 1
                        filtered_samples.append(sample_path)

                all_samples = filtered_samples
                print(
                    f"{self.blur_term} Exclusion Filter: Excluded {excluded_count} samples with |{self.blur_term}| < {min_blur_filter} px")
                if missing_metadata_count > 0:
                    print(
                        f"  WARNING: {missing_metadata_count} samples missing from metadata CSV (included in test)")
            else:
                print(
                    f"  WARNING: Metadata CSV not found at {metadata_csv} - skipping {self.blur_term} exclusion filter")

        print(f"Testing on {len(all_samples)} samples (out of {total_available} available)")
        print(f"Batch size: {batch_size}, Num workers: {num_workers}")

        # Calculate visualization samples based on percentage
        save_viz_indices = set()
        if output_dir:
            n = max(1, len(all_samples))
            # Calculate number of visualizations based on percentage
            num_viz = max(1, int(round(n * viz_percent / 100.0)))
            # Select random indices for variety across test runs
            import random
            save_viz_indices = set(random.sample(range(n), min(num_viz, n)))
            print(
                f"Visualization saving: {len(save_viz_indices)} comparisons ({viz_percent}% of {n} samples)")

        dataset = TestDataset(all_samples, data_dir, self.max_blur)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == 'cuda')
        )

        results = []
        errors = []

        # Store data for visual samples
        visual_samples_data = []

        global_idx = 0  # Track global sample index across batches

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Testing DME"):
                # Move batch to device
                blur_batch = batch['blur'].to(self.device)
                blur_gt_batch = batch['blur_map'].to(self.device)
                blur_values_gt = batch['blur_value']
                sample_names = batch['sample_name']

                # Predict blur maps for entire batch
                pred_blur_map_batch = self.model.dme_subnet(blur_batch)

                # Process each sample in the batch
                for j in range(blur_batch.size(0)):
                    blur = blur_batch[j:j + 1]
                    blur_gt = blur_gt_batch[j:j + 1]
                    pred_blur_map = pred_blur_map_batch[j:j + 1]
                    blur_value_gt = blur_values_gt[j]
                    sample_name = sample_names[j]

                    if isinstance(blur_value_gt, torch.Tensor):
                        blur_value_gt = blur_value_gt.cpu().item()
                    else:
                        blur_value_gt = float(blur_value_gt)

                    pred_blur_px = self.denormalize_blur(pred_blur_map.mean())
                    error = abs(pred_blur_px - blur_value_gt)
                    errors.append(error)

                    # Convert blur to defocus distance (mm)
                    # Note: Returns magnitude only, sign ambiguity is inherent to blur measurement
                    defocus_gt_mm = self.blur_calc.blur_to_defocus(blur_value_gt)
                    defocus_pred_mm = self.blur_calc.blur_to_defocus(pred_blur_px)
                    defocus_error_mm = abs(defocus_pred_mm - defocus_gt_mm)
                    defocus_error_pct = (defocus_error_mm / abs(defocus_gt_mm)
                                         * 100) if defocus_gt_mm != 0 else 0

                    # Compute calibration uncertainty for this prediction
                    unc_mm = 0.0
                    if self.rho_std > 0 and self.rho_direct > 0:
                        unc_mm = defocus_uncertainty(
                            pred_blur_px, self.rho_direct, self.sigma_0_direct,
                            self.rho_std, self.sigma_0_std,
                        )

                    results.append({
                        'sample': sample_name,
                        f'{self.blur_col}_gt_px': blur_value_gt,
                        f'{self.blur_col}_pred_px': pred_blur_px,
                        'error_px': error,
                        'error_pct': (error / blur_value_gt * 100) if blur_value_gt > 0 else 0,
                        'defocus_gt_mm': defocus_gt_mm,
                        'defocus_pred_mm': defocus_pred_mm,
                        'defocus_error_mm': defocus_error_mm,
                        'defocus_error_pct': defocus_error_pct,
                        'defocus_uncertainty_mm': unc_mm,
                    })

                    # Debug: Print first 3 predictions to verify model is running
                    if global_idx < 3:
                        print(
                            f"Sample {global_idx}: GT={blur_value_gt:.2f}px, Pred={pred_blur_px:.2f}px, Error={error:.2f}px")

                    # Store data for visualization
                    if (output_dir is not None) and (global_idx in save_viz_indices):
                        visual_samples_data.append({
                            'sample_path': all_samples[global_idx],
                            'blur': blur.cpu(),
                            'blur_gt': blur_gt.cpu(),
                            'pred_blur_map': pred_blur_map.cpu(),
                            'blur_value_gt': blur_value_gt,
                            'pred_blur_px': pred_blur_px,
                            'error': error,
                            'defocus_uncertainty_mm': unc_mm,
                        })

                    global_idx += 1

        df = pd.DataFrame(results)

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        median_error = np.median(errors)
        std_error = np.std(errors)

        # Filtered metrics (exclude low blur samples if filter_metrics enabled)
        # Only show this comparison if samples weren't already excluded from testing
        if filter_metrics and min_blur_filter is not None and not exclude_from_test:
            df_filtered = df[df[f'{self.blur_col}_gt_px'].abs() >= min_blur_filter].copy()

            if len(df_filtered) > 0:
                filtered_errors = df_filtered['error_px'].values
                filtered_mae = np.mean(filtered_errors)
                filtered_rmse = np.sqrt(np.mean(filtered_errors ** 2))
                filtered_median = np.median(filtered_errors)
                filtered_std = np.std(filtered_errors)

                print(f"\n{'='*60}")
                print(f"Filtered Metrics ({self.blur_term} >= {min_blur_filter} px)")
                print(f"{'='*60}")
                print(
                    f"  Samples: {len(df_filtered)} ({len(df_filtered)/len(df)*100:.1f}% of total)")
                print(
                    f"  Excluded: {len(df) - len(df_filtered)} low-{self.blur_term} samples where metrics are less meaningful")
                print(f"\n  MAE:    {filtered_mae:.2f} px (vs {mae:.2f} px full dataset)")
                print(f"  RMSE:   {filtered_rmse:.2f} px (vs {rmse:.2f} px full dataset)")
                print(f"  Median: {filtered_median:.2f} px (vs {median_error:.2f} px full dataset)")
                print(f"  Std:    {filtered_std:.2f} px (vs {std_error:.2f} px full dataset)")

                # Compute binned metrics for filtered data
                max_blur_ceil = int(np.ceil(self.max_blur))
                bin_range = max_blur_ceil - min_blur_filter
                num_bins = 4
                bin_size = bin_range / num_bins
                filtered_bins = [
                    (min_blur_filter + i * bin_size, min_blur_filter + (i + 1) * bin_size)
                    for i in range(num_bins)]

                bin_weights = self.bin_weights
                filtered_bin_maes = []
                filtered_bin_counts = []

                for low, high in filtered_bins:
                    mask = (df_filtered[f'{self.blur_col}_gt_px'] >= low) & (
                        df_filtered[f'{self.blur_col}_gt_px'] < high)
                    bin_errors = df_filtered[mask]['error_px'].values
                    if len(bin_errors) > 0:
                        filtered_bin_maes.append(np.mean(bin_errors))
                        filtered_bin_counts.append(len(bin_errors))
                    else:
                        filtered_bin_maes.append(0.0)
                        filtered_bin_counts.append(0)

                # Compute weighted MAE for filtered data
                filtered_weighted_mae = sum(w * m for w, m in zip(bin_weights, filtered_bin_maes))

                # Print filtered binned metrics
                weights_str = '-'.join([f"{int(w*100)}" for w in bin_weights])
                print(f"\n  Binned MAE (filtered, weighted {weights_str}%):")
                filtered_bin_labels = [f"{low:.1f}-{high:.1f}" for low, high in filtered_bins]
                for label, bin_mae, count, weight in zip(
                        filtered_bin_labels, filtered_bin_maes, filtered_bin_counts, bin_weights):
                    print(f"    {label} px: {bin_mae:.2f} px (n={count}, weight={weight*100:.0f}%)")
                print(f"  Filtered Weighted MAE: {filtered_weighted_mae:.2f} px")
            else:
                print(
                    f"\n⚠ No samples with {self.blur_term} >= {min_blur_filter} px for filtered metrics")

        # Compute binned MAE (aligned with training)
        # If filtering is enabled, bins start from min_blur_filter instead of 0
        if min_blur_filter is not None and min_blur_filter > 0:
            # Filtered bins: compute bins from min_blur to max_blur
            max_blur_ceil = int(np.ceil(self.max_blur))
            bin_range = max_blur_ceil - min_blur_filter
            num_bins = 4
            bin_size = bin_range / num_bins
            bins = [
                (min_blur_filter + i * bin_size, min_blur_filter + (i + 1) * bin_size)
                for i in range(num_bins)]
        else:
            # Standard bins: 0 to max_blur
            bins = self._get_bins()

        bin_weights = self.bin_weights
        bin_maes = []
        bin_counts = []

        for low, high in bins:
            mask = (df[f'{self.blur_col}_gt_px'] >= low) & (df[f'{self.blur_col}_gt_px'] < high)
            bin_errors = df[mask]['error_px'].values
            if len(bin_errors) > 0:
                bin_maes.append(np.mean(bin_errors))
                bin_counts.append(len(bin_errors))
            else:
                bin_maes.append(0.0)
                bin_counts.append(0)

        weighted_mae = sum(w * m for w, m in zip(bin_weights, bin_maes))

        bin_labels = [f"{int(low)}-{int(high)}" for low, high in bins]

        defocus_mae_mm = df['defocus_error_mm'].mean()
        defocus_rmse_mm = np.sqrt((df['defocus_error_mm']**2).mean())
        defocus_median_mm = df['defocus_error_mm'].median()

        print(f"\nDME Testing Results:")
        print(f"  {self.blur_term} Metrics:")
        print(f"    Uniform MAE:  {mae:.2f} px")
        print(f"    Weighted MAE: {weighted_mae:.2f} px")
        print(f"    RMSE:         {rmse:.2f} px")
        print(f"    Median:       {median_error:.2f} px")
        print(f"    Std:          {std_error:.2f} px")
        print(f"  Defocus Distance Metrics:")
        print(f"    MAE:    {defocus_mae_mm:.2f} mm")
        print(f"    RMSE:   {defocus_rmse_mm:.2f} mm")
        print(f"    Median: {defocus_median_mm:.2f} mm")
        print(f"\n{'='*60}")
        weights_str = '-'.join([f"{int(w*100)}" for w in self.bin_weights])
        print(f"{self.blur_term} Binned Metrics (weighted {weights_str}%)")
        if min_blur_filter is not None and min_blur_filter > 0:
            print(f"Filtered range: {min_blur_filter}-{max_blur_ceil} px")
        print(f"{'='*60}")
        print(f"\nBinned MAE:")
        for i, (label, bin_mae, count) in enumerate(zip(bin_labels, bin_maes, bin_counts)):
            print(f"  {label} px: {bin_mae:.2f} px (n={count}, weight={bin_weights[i]*100:.0f}%)")

        # Per-integer interval analysis (unweighted) - ALWAYS computed
        # If filtering is enabled, intervals start from min_blur_filter instead of 0
        max_blur_ceil = int(np.ceil(self.max_blur))

        if min_blur_filter is not None and min_blur_filter > 0:
            # Filtered intervals: start from ceil(min_blur_filter) to max_blur
            start_interval = int(np.ceil(min_blur_filter))
            intervals = [(i, i + 1) for i in range(start_interval, max_blur_ceil)]
        else:
            # Standard intervals: 0 to max_blur
            intervals = [(i, i + 1) for i in range(max_blur_ceil)]

        interval_maes = []
        interval_counts = []
        interval_labels = []

        for low, high in intervals:
            mask = (df[f'{self.blur_col}_gt_px'] >= low) & (df[f'{self.blur_col}_gt_px'] < high)
            interval_df = df[mask]

            if len(interval_df) > 0:
                interval_maes.append(interval_df['error_px'].mean())
                interval_counts.append(len(interval_df))
                interval_labels.append(f"{low}-{high}")
            else:
                interval_maes.append(np.nan)
                interval_counts.append(0)
                interval_labels.append(f"{low}-{high}")

        # Print per-integer metrics
        print(f"\n{'='*60}")
        print(f"Per-Integer Interval Metrics (unweighted)")
        if min_blur_filter is not None and min_blur_filter > 0:
            print(
                f"Filtered range: {start_interval}-{max_blur_ceil} px (starting from ceil({min_blur_filter})={start_interval})")
        print(f"{'='*60}")

        print(f"\nPer-Integer MAE:")
        for label, mae, count in zip(interval_labels, interval_maes, interval_counts):
            if not np.isnan(mae):
                print(f"  {label} px: {mae:.2f} px (n={count})")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if create_subfolder:
                # CLI / generic-output-dir mode: per-test slot inside output_dir
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dme_test_dir = output_dir / 'DME_test' / f'test_{timestamp}'
                dme_test_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Caller has already constructed a per-test folder — write into it directly
                dme_test_dir = output_dir

            csv_path = dme_test_dir / 'dme_test_results.csv'
            df.to_csv(csv_path, index=False)
            print(f"\nSaved results to: {csv_path}")

            config = {
                'data_dir': str(data_dir),
                'model_path': self.model_path,
                'device': str(self.device),
                'num_samples': num_samples,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'viz_percent': viz_percent,
                'num_worst_px': num_worst_px,
                'num_worst_pct': num_worst_pct,
                'min_coc_filter': min_blur_filter,  # config key kept for compat
                'filter_worst_pct': filter_worst_pct,
                'filter_metrics': filter_metrics,
                'exclude_from_test': exclude_from_test
            }
            self._save_test_summary(dme_test_dir, 'DME', df, config)

            self._plot_dme_results(df, errors, dme_test_dir, min_blur_filter)

            self._create_dme_visual_comparisons(
                visual_samples_data, dme_test_dir, num_visual_samples, csv_path, data_dir)

            if num_worst_px > 0 or num_worst_pct > 0:
                self._save_dme_worst_cases(
                    df, data_dir, dme_test_dir,
                    num_worst_px=num_worst_px, num_worst_pct=num_worst_pct,
                    min_blur_filter=min_blur_filter if filter_worst_pct else None)

        return df

    def _save_test_summary(
        self,
        output_dir: Path,
        test_mode: str,
        df: pd.DataFrame,
        config: dict
    ):
        """
        Save a comprehensive test summary with all settings and results.

        Args:
            output_dir: Directory to save summary file
            test_mode: 'DME' or 'Dual'
            df: Results DataFrame
            config: Dictionary with test configuration parameters
        """
        summary_path = output_dir / 'test_summary.txt'

        with open(summary_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write(f"{test_mode} Test Summary\n")
            f.write("=" * 80 + "\n\n")

            # Timestamp
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Directory: {output_dir}\n\n")

            # Test Configuration
            f.write("-" * 80 + "\n")
            f.write("TEST CONFIGURATION\n")
            f.write("-" * 80 + "\n")

            # Model identity (derived from path) — explicit in the summary so
            # readers don't need to parse the directory tree to know which
            # model + variant produced these results.
            try:
                from run_paths import detect_model_name, detect_variant
                model_name = detect_model_name(output_dir)
                variant = detect_variant(output_dir)
                if model_name:
                    f.write(f"Model Name: {model_name}\n")
                    f.write(f"Variant: {variant}\n")
            except Exception:
                pass

            f.write(f"Data Directory: {config.get('data_dir', 'N/A')}\n")
            f.write(f"Model Path: {config.get('model_path', 'N/A')}\n")
            f.write(f"Device: {config.get('device', 'N/A')}\n")
            f.write(f"Max {self.blur_term}: {self.max_blur:.2f} px\n\n")

            f.write(f"Samples Requested: {config.get('num_samples', 'N/A')}\n")
            f.write(f"Samples Tested: {len(df)}\n")
            f.write(f"Batch Size: {config.get('batch_size', 'N/A')}\n")
            f.write(f"Num Workers: {config.get('num_workers', 'N/A')}\n\n")

            # Filtering Settings
            f.write(f"{self.blur_term} Filtering:\n")
            min_blur_filter = config.get('min_coc_filter')  # config key kept for compat
            if min_blur_filter is not None:
                f.write(f"  Enabled: Yes\n")
                f.write(f"  Min {self.blur_term} Threshold: {min_blur_filter} px\n")
                f.write(f"  Filter % Error Worst Cases: {config.get('filter_worst_pct', False)}\n")
                f.write(f"  Show Filtered Metrics: {config.get('filter_metrics', False)}\n")
                f.write(f"  Exclude from Testing: {config.get('exclude_from_test', False)}\n")
            else:
                f.write(f"  Enabled: No\n")
            f.write("\n")

            # Visualization Settings
            f.write("Visualization:\n")
            f.write(f"  Viz Percentage: {config.get('viz_percent', 'N/A')}%\n")
            f.write(f"  Worst Cases (px): {config.get('num_worst_px', 0)}\n")
            # Different worst case % metric depending on test mode
            if test_mode == 'DME':
                f.write(f"  Worst Cases ({self.blur_term} %): {config.get('num_worst_pct', 0)}\n\n")
            else:  # Dual
                f.write(f"  Worst Cases (defocus %): {config.get('num_worst_defocus_pct', 0)}\n\n")

            # Overall Results
            f.write("-" * 80 + "\n")
            f.write("OVERALL RESULTS\n")
            f.write("-" * 80 + "\n")

            if test_mode == 'DME':
                errors = df['error_px'].values
                f.write(f"{self.blur_term} MAE: {np.mean(errors):.4f} px\n")
                f.write(f"{self.blur_term} RMSE: {np.sqrt(np.mean(errors**2)):.4f} px\n")
                f.write(f"{self.blur_term} Median Error: {np.median(errors):.4f} px\n")
                f.write(f"{self.blur_term} Std Dev: {np.std(errors):.4f} px\n")
                f.write(f"{self.blur_term} 95th Percentile: {np.percentile(errors, 95):.4f} px\n")

                # Binned metrics if available
                if f'{self.blur_col}_gt_px' in df.columns:
                    f.write("\n")
                    weights_str = '-'.join([f"{int(w*100)}" for w in self.bin_weights])
                    f.write(f"Binned Metrics (4 bins, weighted {weights_str}%):\n")
                    bins = self._get_bins()
                    bin_weights = self.bin_weights

                    for (low, high), weight in zip(bins, bin_weights):
                        mask = (df[f'{self.blur_col}_gt_px'] >= low) & (
                            df[f'{self.blur_col}_gt_px'] < high)
                        bin_df = df[mask]
                        if len(bin_df) > 0:
                            bin_mae = bin_df['error_px'].mean()
                            f.write(
                                f"  {int(low)}-{int(high)} px: MAE={bin_mae:.4f} px (n={len(bin_df)}, weight={weight*100:.0f}%)\n")

            elif test_mode == 'Dual':
                f.write(f"PSNR: {df['psnr_db'].mean():.2f} dB\n")
                f.write(f"SSIM: {df['ssim'].mean():.4f}\n")
                f.write(f"Sharp MAE: {df['sharp_mae'].mean():.2f} px\n")
                f.write(f"Sharp RMSE: {df['sharp_rmse'].mean():.2f} px\n\n")

                # Include defocus distance alongside blur metric
                defocus_mae = df['defocus_error_mm'].mean(
                ) if 'defocus_error_mm' in df.columns else 0
                f.write(
                    f"{self.blur_term} Error: {df[f'{self.blur_col}_error_px'].mean():.4f} px ({defocus_mae:.2f} mm)\n")
                if 'blur_map_mae_px' in df.columns:
                    f.write(f"{self.blur_term} Map MAE: {df['blur_map_mae_px'].mean():.4f} px\n")
                # Compute RMSE from error column
                blur_rmse = np.sqrt((df[f'{self.blur_col}_error_px']**2).mean())
                defocus_rmse = np.sqrt((df['defocus_error_mm']**2).mean()
                                       ) if 'defocus_error_mm' in df.columns else 0
                f.write(f"{self.blur_term} RMSE: {blur_rmse:.4f} px ({defocus_rmse:.2f} mm)\n\n")

                f.write(f"Diameter Error: {df['diameter_error_px'].mean():.2f} px\n")
                f.write(f"Diameter Error %: {df['diameter_error_pct'].mean():.2f}%\n")
                # Compute Diameter RMSE from error column
                diameter_rmse = np.sqrt((df['diameter_error_px']**2).mean())
                f.write(f"Diameter RMSE: {diameter_rmse:.2f} px\n")
                if 'grad_mae' in df.columns:
                    f.write(f"Gradient MAE: {df['grad_mae'].mean():.4f}\n")

            f.write("\n")

            # Filtered metrics if applicable
            if min_blur_filter is not None and config.get('filter_metrics', False):
                df_filtered = df[df[f'{self.blur_col}_gt_px'].abs() >= min_blur_filter].copy()
                if len(df_filtered) > 0:
                    f.write("-" * 80 + "\n")
                    f.write(f"FILTERED RESULTS ({self.blur_term} >= {min_blur_filter} px)\n")
                    f.write("-" * 80 + "\n")
                    f.write(
                        f"Filtered Samples: {len(df_filtered)} ({len(df_filtered)/len(df)*100:.1f}% of total)\n")
                    f.write(f"Excluded Samples: {len(df) - len(df_filtered)}\n\n")

                    if test_mode == 'DME':
                        filtered_errors = df_filtered['error_px'].values
                        f.write(f"{self.blur_term} MAE: {np.mean(filtered_errors):.4f} px\n")
                        f.write(f"{self.blur_term} RMSE: {np.sqrt(np.mean(filtered_errors**2)):.4f} px\n")
                    elif test_mode == 'Dual':
                        f.write(f"PSNR: {df_filtered['psnr_db'].mean():.2f} dB\n")
                        f.write(f"SSIM: {df_filtered['ssim'].mean():.4f}\n")
                        filtered_defocus = df_filtered['defocus_error_mm'].mean(
                        ) if 'defocus_error_mm' in df_filtered.columns else 0
                        f.write(
                            f"{self.blur_term} Error: {df_filtered[f'{self.blur_col}_error_px'].mean():.4f} px ({filtered_defocus:.2f} mm)\n")
                        f.write(
                            f"Diameter Error: {df_filtered['diameter_error_px'].mean():.2f} px\n")
                        f.write(
                            f"Diameter Error %: {df_filtered['diameter_error_pct'].mean():.2f}%\n")
                    f.write("\n")

            # File outputs
            f.write("-" * 80 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Results CSV: {output_dir / (test_mode.lower() + '_test_results.csv')}\n")
            f.write(f"Analysis Plots: {output_dir / (test_mode.lower() + '_analysis.png')}\n")

            if test_mode == 'DME':
                f.write(f"Visual Comparisons: {output_dir / 'dme_visual_comparisons'}\n")
                f.write(
                    f"Grid Summary: {output_dir / 'dme_visual_comparisons' / 'dme_grid_summary.png'}\n")
                if config.get('num_worst_px', 0) > 0 or config.get('num_worst_pct', 0) > 0:
                    f.write(f"Worst Cases: {output_dir / 'worst_cases_dme'}\n")
            elif test_mode == 'Dual':
                f.write(f"Visualizations: {output_dir / 'visualizations'}\n")
                f.write(f"Dual Analysis: {output_dir / 'dual_analysis.png'}\n")
                if config.get('num_worst_px', 0) > 0 or config.get('num_worst_defocus_pct', 0) > 0:
                    f.write(f"Worst Cases: {output_dir / 'worst_cases'}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Saved test summary to: {summary_path}")

    def _plot_dme_results(
        self,
        df: pd.DataFrame,
        errors: List[float],
        output_dir: Path,
        min_blur_filter: Optional[float] = None
    ):
        """Create visualization plots for DME results."""

        # Error distribution - 2 rows x 4 columns
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Row 0: Blur Error Analysis (4 plots across)
        # 1. Error histogram
        axes[0, 0].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Absolute Error (px)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'{self.blur_term} Error Distribution')
        axes[0, 0].axvline(np.mean(errors), color='r', linestyle='--',
                           label=f'MAE: {np.mean(errors):.2f}')
        axes[0, 0].legend()

        # 2. Predicted vs Reference scatter
        axes[0, 1].scatter(df[f'{self.blur_col}_gt_px'],
                           df[f'{self.blur_col}_pred_px'], alpha=0.5, s=20)
        axes[0, 1].plot([0, df[f'{self.blur_col}_gt_px'].max()], [
                        0, df[f'{self.blur_col}_gt_px'].max()], 'r--', label='Perfect')
        axes[0, 1].set_xlabel(f'Reference {self.blur_term} (px)')
        axes[0, 1].set_ylabel(f'Predicted {self.blur_term} (px)')
        axes[0, 1].set_title(f'Predicted vs Reference {self.blur_term}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Cumulative error distribution
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        axes[0, 2].plot(sorted_errors, cumulative, linewidth=2)
        axes[0, 2].set_xlabel('Absolute Error (px)')
        axes[0, 2].set_ylabel('Cumulative %')
        axes[0, 2].set_title(f'Cumulative {self.blur_term} Error')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axhline(50, color='r', linestyle='--', alpha=0.5, label='50th')
        axes[0, 2].axhline(95, color='g', linestyle='--', alpha=0.5, label='95th')

        # Add percentile annotations
        p50_error = np.percentile(errors, 50)
        p95_error = np.percentile(errors, 95)
        axes[0, 2].text(0.98, 0.50, f'{p50_error:.2f}px', transform=axes[0, 2].get_yaxis_transform(
        ), ha='right', va='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[0, 2].text(0.98, 0.95, f'{p95_error:.2f}px', transform=axes[0, 2].get_yaxis_transform(
        ), ha='right', va='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[0, 2].legend(loc='lower right', fontsize=8)

        # 4. Error vs Reference scatter
        axes[0, 3].scatter(df[f'{self.blur_col}_gt_px'], df['error_px'], alpha=0.5, s=20)
        axes[0, 3].set_xlabel(f'Reference {self.blur_term} (px)')
        axes[0, 3].set_ylabel('Absolute Error (px)')
        axes[0, 3].set_title(f'Error vs Reference {self.blur_term}')
        axes[0, 3].grid(True, alpha=0.3)

        # Row 1: Binned Blur + Defocus Distance Analysis (4 plots across)
        # 5. Binned MAE (weighted)
        if min_blur_filter is not None and min_blur_filter > 0:
            # Filtered bins: compute bins from min_blur to max_blur
            max_blur_ceil = int(np.ceil(self.max_blur))
            bin_range = max_blur_ceil - min_blur_filter
            num_bins = 4
            bin_size = bin_range / num_bins
            bins = [
                (min_blur_filter + i * bin_size, min_blur_filter + (i + 1) * bin_size)
                for i in range(num_bins)]
        else:
            # Standard bins: 0 to max_blur
            bins = self._get_bins()

        bin_weights = self.bin_weights
        bin_maes = []
        bin_counts = []

        for low, high in bins:
            mask = (df[f'{self.blur_col}_gt_px'] >= low) & (df[f'{self.blur_col}_gt_px'] < high)
            bin_errors = df[mask]['error_px'].values
            if len(bin_errors) > 0:
                bin_maes.append(np.mean(bin_errors))
                bin_counts.append(len(bin_errors))
            else:
                bin_maes.append(0.0)
                bin_counts.append(0)

        bin_labels = [f"{int(low)}-{int(high)}" for low, high in bins]
        x_pos = np.arange(len(bin_labels))

        bars = axes[1, 0].bar(x_pos, bin_maes, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel(f'{self.blur_term} Range (px)')
        axes[1, 0].set_ylabel('MAE (px)')
        weights_str = '-'.join([f"{int(w*100)}" for w in self.bin_weights])
        axes[1, 0].set_title(f'Binned MAE ({weights_str}% weights)')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(bin_labels, fontsize=8)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Add count labels on bars
        for i, (bar, count, mae) in enumerate(zip(bars, bin_counts, bin_maes)):
            height = bar.get_height()
            if height > 0:
                axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height,
                                f'{mae:.2f}\n(n={count})',
                                ha='center', va='bottom', fontsize=8)

        # Add padding above the tallest bar to fit the labels
        if len(bin_maes) > 0 and max(bin_maes) > 0:
            axes[1, 0].set_ylim(top=max(bin_maes) * 1.15)

        # 6. Defocus Distance Error Distribution
        defocus_errors = df['defocus_error_mm'].values
        axes[1, 1].hist(defocus_errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1, 1].set_xlabel('Defocus Error (mm)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Defocus Error Distribution')
        defocus_mae = np.mean(defocus_errors)
        axes[1, 1].axvline(defocus_mae, color='r', linestyle='--',
                           label=f'MAE: {defocus_mae:.2f} mm')
        axes[1, 1].legend()

        # 7. Predicted vs Reference Defocus Distance
        axes[1, 2].scatter(df['defocus_gt_mm'], df['defocus_pred_mm'],
                           alpha=0.5, s=20, color='steelblue')
        max_defocus = max(df['defocus_gt_mm'].abs().max(), df['defocus_pred_mm'].abs().max())
        axes[1, 2].plot([0, max_defocus], [0, max_defocus], 'r--', label='Perfect')
        axes[1, 2].set_xlabel('Reference Defocus (mm)')
        axes[1, 2].set_ylabel('Predicted Defocus (mm)')
        axes[1, 2].set_title('Predicted vs Reference Defocus')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        # 8. Error vs Reference Defocus Distance
        axes[1, 3].scatter(df['defocus_gt_mm'], df['defocus_error_mm'],
                           alpha=0.5, s=20, color='steelblue')
        axes[1, 3].set_xlabel('Reference Defocus (mm)')
        axes[1, 3].set_ylabel('Defocus Error (mm)')
        axes[1, 3].set_title('Error vs Reference Defocus')
        axes[1, 3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'dme_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved plots to: {output_dir / 'dme_analysis.png'}")

    def _create_dme_visual_comparisons(
        self,
        samples_data: List[Dict],
        output_dir: Path,
        num_samples: int = 16,
        csv_path: Path = None,
        data_dir: Path = None
    ):
        """
        Create visual comparison images for DME results.

        Shows: Blurred Input | Reference Blur Map | Predicted Blur Map | Error Map

        Note: Reference blur is relative to sharpest preprocessing crops, not absolute ground truth.
        Selects samples spread across the blur range to show performance at different defocus levels.
        """
        if len(samples_data) == 0:
            return

        # Use ALL samples from samples_data (already filtered by viz_percent in test loop)
        # No need to subsample again
        selected_samples = list(samples_data)
        n = len(selected_samples)

        vis_dir = output_dir / 'dme_visual_comparisons'
        vis_dir.mkdir(exist_ok=True)

        print(f"\nGenerating {n} visual comparisons...")

        for sample in tqdm(selected_samples, desc="Creating visuals"):
            self._plot_dme_sample(sample, vis_dir)

        # Create grid summary from CSV (5x5 = 25 samples evenly distributed across blur range)
        if csv_path and csv_path.exists() and data_dir:
            self._create_dme_grid_summary_from_csv(csv_path, vis_dir, data_dir, num_grid_samples=25)

        print(f"Saved visual comparisons to: {vis_dir}")

    def _save_dme_worst_cases(
        self, df: pd.DataFrame, data_dir: Path, output_dir: Path,
        num_worst_px: int = 10, num_worst_pct: int = 10,
        min_blur_filter: Optional[float] = None,
    ):
        """Save side-by-side visualisations of the N worst-performing samples.

        Two folders, one for each ranking metric:
            <output_dir>/worst_cases_dme/by_blur_error_px/
            <output_dir>/worst_cases_dme/by_blur_error_pct/

        ``min_blur_filter`` (when set) excludes samples with |blur_gt| below
        that threshold from the percentage ranking, since % error is
        meaningless for very-small ground-truth blurs.
        """
        if len(df) == 0:
            return

        worst_cases_dir = output_dir / 'worst_cases_dme'
        worst_cases_dir.mkdir(exist_ok=True)
        print("\nSaving worst-case visualisations...")

        def _render(row, save_path: Path, title_extra: str):
            sample_name = row['sample']
            blur_gt = row[f'{self.blur_col}_gt_px']
            blur_pred = row[f'{self.blur_col}_pred_px']
            defocus_gt = row['defocus_gt_mm']
            defocus_pred = row['defocus_pred_mm']

            blur_path = data_dir / 'blur' / f"{sample_name}.png"
            bm_dir = data_dir / 'blur_map' if (data_dir / 'blur_map').exists() \
                else data_dir / 'coc_map'
            blur_map_path = bm_dir / f"{sample_name}.png"
            if not blur_path.exists() or not blur_map_path.exists():
                print(f"  Warning: skipping {sample_name} (files not found)")
                return

            blur_img = cv2.imread(str(blur_path), cv2.IMREAD_GRAYSCALE)
            blur_map_gt = cv2.imread(str(blur_map_path), cv2.IMREAD_GRAYSCALE)
            blur_map_gt_norm = (blur_map_gt.astype(np.float32) / 255.0) * 2 - 1
            pred_normalized = (blur_pred / self.max_blur) * 2.0 - 1.0
            pred_uniform = np.full_like(blur_map_gt_norm, pred_normalized,
                                        dtype=np.float32)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(blur_img, cmap='gray')
            axes[0].set_title("Blur (Input)")
            axes[0].axis('off')
            im1 = axes[1].imshow(blur_map_gt_norm, cmap='RdYlGn_r',
                                 vmin=-1, vmax=1)
            axes[1].set_title(
                f"Ground Truth {self.blur_term}\n"
                f"{blur_gt:.2f} px ({defocus_gt:.2f} mm)")
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            im2 = axes[2].imshow(pred_uniform, cmap='RdYlGn_r',
                                 vmin=-1, vmax=1)
            axes[2].set_title(
                f"Predicted {self.blur_term}\n"
                f"{blur_pred:.2f} px ({defocus_pred:.2f} mm)")
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            plt.suptitle(f"Worst Case: {sample_name}\n{title_extra}",
                         fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        # Worst by absolute blur error (px)
        if num_worst_px > 0:
            sub = worst_cases_dir / 'by_blur_error_px'
            sub.mkdir(exist_ok=True)
            worst_px = df.sort_values('error_px', ascending=False).head(num_worst_px)
            print(f"  Saving {len(worst_px)} worst by {self.blur_term} error px → {sub.name}/")
            for _, row in worst_px.iterrows():
                title = (f"{self.blur_term} Error: {row['error_px']:.2f} px "
                         f"({row['defocus_error_mm']:.2f} mm)")
                fname = (f"{row['sample']}_BlurErr{row['error_px']:.2f}px"
                         f"_{row['defocus_error_mm']:.2f}mm.png")
                _render(row, sub / fname, title)

        # Worst by percentage error
        if num_worst_pct > 0:
            sub = worst_cases_dir / 'by_blur_error_pct'
            sub.mkdir(exist_ok=True)
            if min_blur_filter is not None:
                df_filt = df[df[f'{self.blur_col}_gt_px'].abs() >= min_blur_filter].copy()
                if len(df_filt) == 0:
                    print(f"  Warning: no samples with |{self.blur_term}| >= "
                          f"{min_blur_filter} px — skipping percentage worst cases")
                    return
                worst_pct = df_filt.sort_values('error_pct', ascending=False).head(num_worst_pct)
                excl = len(df) - len(df_filt)
                print(f"  Saving {len(worst_pct)} worst by {self.blur_term} error % "
                      f"(filtered |{self.blur_term}| ≥ {min_blur_filter} px, "
                      f"excluded {excl}) → {sub.name}/")
            else:
                worst_pct = df.sort_values('error_pct', ascending=False).head(num_worst_pct)
                print(f"  Saving {len(worst_pct)} worst by {self.blur_term} error % "
                      f"→ {sub.name}/")
            for _, row in worst_pct.iterrows():
                title = (f"{self.blur_term} Error: {row['error_pct']:.1f}% "
                         f"({row['error_px']:.2f} px / {row['defocus_error_mm']:.2f} mm)")
                fname = (f"{row['sample']}_BlurErr{row['error_pct']:.1f}pct"
                         f"_{row['defocus_error_mm']:.2f}mm.png")
                _render(row, sub / fname, title)

        if num_worst_px > 0 or num_worst_pct > 0:
            print(f"Saved worst cases to: {worst_cases_dir}")

    def _plot_dme_sample(self, sample_data: Dict, output_dir: Path):
        """Create a single sample comparison image - scalar blur focused."""
        sample_name = sample_data['sample_path'].stem

        # Extract tensor and convert to numpy
        blur = sample_data['blur'].squeeze().numpy()
        blur = (blur + 1.0) / 2.0  # Denormalize to [0, 1]

        gt_blur = sample_data['blur_value_gt']
        pred_blur = sample_data['pred_blur_px']
        error = sample_data['error']

        gt_defocus_mm = self.blur_calc.blur_to_defocus(gt_blur)
        pred_defocus_mm = self.blur_calc.blur_to_defocus(pred_blur)
        defocus_error_mm = abs(pred_defocus_mm - gt_defocus_mm)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1. Blurred input with annotations
        axes[0].imshow(blur, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Blurred Input\n(Sample {sample_name})', fontsize=11)
        axes[0].axis('off')

        # 2. Bar chart comparison
        # Reference is always blue, Predicted is color-coded by % error
        # Green: < 25%, Orange: 25-50%, Red: > 50%
        error_pct = (error / gt_blur * 100) if gt_blur > 0 else 0
        if error_pct < 25:
            pred_color = '#2ecc71'  # Green
        elif error_pct < 50:
            pred_color = '#e67e22'  # Orange
        else:
            pred_color = '#e74c3c'  # Red

        bars = axes[1].bar(['Reference', 'Predicted'], [gt_blur, pred_blur],
                           color=['#3498db', pred_color], edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel(f'{self.blur_term} (px)', fontsize=11)
        axes[1].set_title(
            f'{self.blur_term} Estimation (relative to sharpest crops)\nError: {error:.2f} px ({error/gt_blur*100:.1f}%)',
            fontsize=11)
        # Set y-axis with 20% headroom for labels
        max_val = max(gt_blur, pred_blur)
        y_max = max_val * 1.2
        axes[1].set_ylim(0, y_max)

        # Add value labels on bars (positioned in the 20% gap)
        for bar, val in zip(bars, [gt_blur, pred_blur]):
            # Position text at 5% above bar height (centered in the 20% gap)
            label_y = bar.get_height() + (y_max - bar.get_height()) * 0.25
            axes[1].text(bar.get_x() + bar.get_width() / 2, label_y,
                         f'{val:.1f} px', ha='center', va='center', fontsize=12, fontweight='bold')

        axes[1].grid(axis='y', alpha=0.3)

        # 3. Defocus Distance Bar chart comparison (same color coding)
        unc_mm = sample_data.get('defocus_uncertainty_mm', 0.0)
        bars_defocus = axes[2].bar(['Reference', 'Predicted'], [gt_defocus_mm, pred_defocus_mm],
                                   color=['#3498db', pred_color], edgecolor='black', linewidth=1.5)

        # Add uncertainty error bar on predicted bar
        if unc_mm > 0:
            pred_bar = bars_defocus[1]
            axes[2].errorbar(
                pred_bar.get_x() + pred_bar.get_width() / 2, pred_defocus_mm,
                yerr=unc_mm, fmt='none', ecolor='black', capsize=5, linewidth=2,
            )

        title_unc = f" (cal. unc. ±{unc_mm:.2f})" if unc_mm > 0 else ""
        axes[2].set_ylabel('Defocus Distance (mm)', fontsize=11)
        axes[2].set_title(
            f'Defocus Distance Estimation\nError: {defocus_error_mm:.2f} mm{title_unc}',
            fontsize=11)

        # Set y-axis with 20% headroom for labels
        max_defocus = max(abs(gt_defocus_mm), abs(pred_defocus_mm)) + unc_mm
        y_max_defocus = max(max_defocus * 1.2, 0.1)
        axes[2].set_ylim(0, y_max_defocus)

        # Add value labels on bars
        for bar, val in zip(bars_defocus, [gt_defocus_mm, pred_defocus_mm]):
            label_y = bar.get_height() + (y_max_defocus - bar.get_height()) * 0.25
            label_text = f'{val:.2f} mm'
            if bar == bars_defocus[1] and unc_mm > 0:
                label_text = f'{val:.2f} ± {unc_mm:.2f} mm'
            axes[2].text(bar.get_x() + bar.get_width() / 2, label_y,
                         label_text, ha='center', va='center', fontsize=11, fontweight='bold')

        axes[2].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{sample_name}_dme_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _create_dme_grid_summary_from_csv(
        self,
        csv_path: Path,
        output_dir: Path,
        data_dir: Path,
        num_grid_samples: int = 25
    ):
        """
        Create a 5x5 grid summary showing samples spread evenly across blur range.
        Reads data from CSV and loads images as needed.

        Args:
            csv_path: Path to the CSV file with test results
            output_dir: Directory to save the grid summary
            data_dir: Path to the data directory containing blur images
            num_grid_samples: Number of samples to select (default: 25 for 5x5 grid)
        """
        # Read CSV (ensure 'sample' column is read as string, not float)
        df = pd.read_csv(csv_path, dtype={'sample': str})
        if len(df) == 0:
            return

        min_blur = df[f'{self.blur_col}_gt_px'].min()
        max_blur = df[f'{self.blur_col}_gt_px'].max()
        blur_range = max_blur - min_blur

        # Calculate interval based on range and number of samples
        interval = blur_range / (num_grid_samples - 1) if num_grid_samples > 1 else 0

        print(
            f"\nSelecting grid samples across {self.blur_term} range: {min_blur:.2f} to {max_blur:.2f} px")
        print(f"Using interval: {interval:.2f} px for {num_grid_samples} samples")

        target_blurs = [min_blur + i * interval for i in range(num_grid_samples)]

        # For each target blur, find the sample with the closest actual blur
        selected_rows = []
        selected_indices = set()

        for target_blur in target_blurs:
            # Find closest sample that hasn't been selected yet
            best_idx = None
            best_diff = float('inf')

            for idx, row in df.iterrows():
                if idx in selected_indices:
                    continue

                actual_blur = row[f'{self.blur_col}_gt_px']
                diff = abs(actual_blur - target_blur)

                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx

            if best_idx is not None:
                selected_rows.append(df.iloc[best_idx])
                selected_indices.add(best_idx)
                actual = df.iloc[best_idx][f'{self.blur_col}_gt_px']
                print(
                    f"  Target: {target_blur:.2f} px → Selected: {actual:.2f} px (diff: {best_diff:.2f} px)")

        print(f"Selected {len(selected_rows)} samples for grid summary")

        nrows = 5
        ncols = 5
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))

        for idx, row in enumerate(selected_rows):
            if idx >= nrows * ncols:
                break

            grid_row = idx // ncols
            grid_col = idx % ncols
            ax = axes[grid_row, grid_col]

            sample_name = row['sample']
            blur_path = data_dir / 'blur' / f'{sample_name}.png'

            try:
                blur_img = Image.open(blur_path)
                blur = np.array(blur_img).astype(np.float32) / 255.0

                ax.imshow(blur, cmap='gray', vmin=0, vmax=1)

                gt = row[f'{self.blur_col}_gt_px']
                pred = row[f'{self.blur_col}_pred_px']
                err = row['error_px']

                # Color code by % error: green < 25%, orange 25-50%, red > 50%
                err_pct = (err / gt * 100) if gt > 0 else 0
                if err_pct < 25:
                    color = '#2ecc71'
                elif err_pct < 50:
                    color = '#f39c12'
                else:
                    color = '#e74c3c'

                ax.set_title(
                    f'Ref: {gt:.1f} → Pred: {pred:.1f} px\nErr: {err:.2f} px ({err_pct:.0f}%)',
                    fontsize=9, color=color, fontweight='bold')
                ax.axis('off')

            except Exception as e:
                print(f"Warning: Could not load image for {sample_name}: {e}")
                ax.axis('off')

        # Hide empty subplots
        for idx in range(len(selected_rows), nrows * ncols):
            grid_row = idx // ncols
            grid_col = idx % ncols
            axes[grid_row, grid_col].axis('off')

        plt.suptitle(
            f'DME Grid Summary: {num_grid_samples} Samples Evenly Distributed Across {self.blur_term} Range',
            fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Add space at top for title

        plt.savefig(output_dir / 'dme_grid_summary.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved grid summary to: {output_dir / 'dme_grid_summary.png'}")


def main():
    parser = argparse.ArgumentParser(description="Test defocus estimation models")
    parser.add_argument('--mode', type=str, choices=['dme', '1'], default='dme',
                        help="Testing mode: 'dme'/'1' for blur estimation only")
    parser.add_argument('--model', '-m', type=str,
                        help="Path to model checkpoint")
    parser.add_argument('--config', '-c', type=str, default=None,
                        help="Path to config YAML (optional if in checkpoint)")
    parser.add_argument('--data', '-d', type=str,
                        help="Path to test data directory")
    parser.add_argument('--samples', '-n', type=int, default=100,
                        help="Number of samples to test (0 for all)")
    parser.add_argument('--output', '-o', type=str, default=None,
                        help="Output directory for results and visualizations")
    parser.add_argument('--device', type=str, default='auto',
                        help="Device: 'cuda', 'cpu', or 'auto'")

    args = parser.parse_args()

    # Map numeric choice to mode name
    if args.mode == '1':
        args.mode = 'dme'

    # Auto-detect checkpoint and data directory if not provided
    if args.model is None:
        # Look for checkpoint in training_output/checkpoints/
        default_checkpoint_dir = Path('training_output/checkpoints')
        if default_checkpoint_dir.exists():
            checkpoint_file = default_checkpoint_dir / 'dme_best.pth'

            if checkpoint_file.exists():
                args.model = str(checkpoint_file)
                print(f"\nAuto-detected checkpoint: {args.model}")
            else:
                print("\n" + "=" * 60)
                args.model = input("Path to model checkpoint: ").strip()
        else:
            print("\n" + "=" * 60)
            args.model = input("Path to model checkpoint: ").strip()

    if args.data is None:
        # Look for data in training_output/synthetic_data/
        default_data_dir = Path('training_output/synthetic_data')
        if default_data_dir.exists() and (default_data_dir / 'blur').exists():
            args.data = str(default_data_dir)
            print(f"Auto-detected data directory: {args.data}")
        else:
            # Fallback to old location
            fallback_data_dir = Path('data/synthetic')
            if fallback_data_dir.exists() and (fallback_data_dir / 'blur').exists():
                args.data = str(fallback_data_dir)
                print(f"Auto-detected data directory: {args.data}")
            else:
                args.data = input("Path to test data directory: ").strip()

    if not args.model or not args.data:
        print("\nError: Model checkpoint and data directory are required.")
        return

    # Count available samples in the data directory
    data_path = Path(args.data)
    blur_dir = data_path / 'blur'
    if blur_dir.exists():
        total_samples = len(list(blur_dir.glob('*.png')))
        print("\n" + "=" * 60)
        print(f"Found {total_samples} samples in dataset")
        samples_input = input(
            f"Number of samples to test (press Enter for all {total_samples}): ").strip()
        if samples_input:
            try:
                requested = int(samples_input)
                if requested > total_samples:
                    print(
                        f"Warning: Requested {requested} but only {total_samples} available. Using {total_samples}.")
                    args.samples = total_samples
                elif requested <= 0:
                    args.samples = total_samples
                else:
                    args.samples = requested
            except ValueError:
                print(f"Invalid number, using all {total_samples} samples")
                args.samples = total_samples
        else:
            args.samples = total_samples
    else:
        print(f"\nWarning: Could not find blur directory at {blur_dir}")
        print(f"Using default: {args.samples} samples")

    tester = ModelTester(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )

    if args.output is None:
        args.output = 'test_results'
        print(f"Output directory: {args.output}")

    # Run test
    df = tester.test_dme_only(
        data_dir=args.data,
        num_samples=args.samples,
        output_dir=args.output
    )

    print("\n✓ Testing complete!")


if __name__ == "__main__":
    main()
