"""
Real Droplet Crop Inference Script

Processes preprocessed droplet crops from cine files and estimates:
- Blur (sigma in direct mode, CoC in optical mode) for defocus depth estimation

Input structure:
    OUTPUT/
        8mm-borosilicate-2/
            sphere0001v_crop.png
            sphere0002v_crop.png
            ...
        6mm-steel-1/
            sphere0001v_crop.png
            ...

Output structure:
    inference_results/
        8mm-borosilicate-2/
            blur_results.csv  (filename, sigma_px/coc_px, defocus_mm)
            visualizations/
                sphere0001v_comparison.png  (side-by-side)
        summary_all_materials.csv
        blur_distribution_by_material.png
"""

import logging
import torch
import torch.nn as nn
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import argparse
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

import sys as _sys
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

from model import DefocusNet
from synthetic_blur import BlurParams, BlurCalculator
from physics import ScalingParams, invert_prediction, defocus_uncertainty


class DiameterMeasurer:
    """Measure droplet diameter from image using automatic Otsu thresholding."""

    def __init__(self, use_otsu: bool = True, fallback_threshold: float = 0.25):
        """
        Args:
            use_otsu: Use Otsu's method for automatic thresholding (recommended)
            fallback_threshold: Manual threshold if Otsu fails or use_otsu=False (0-1)
        """
        self.use_otsu = use_otsu
        self.fallback_threshold = fallback_threshold

    def measure_diameter(self, img: np.ndarray) -> tuple[float, np.ndarray, tuple[float, float]]:
        """
        Measure droplet diameter in pixels using automatic Otsu thresholding.

        Args:
            img: Grayscale image (0-1 range), droplet should be darker than background

        Returns:
            Tuple of (diameter in pixels, binary mask, (center_x, center_y))
        """
        # Convert to uint8 for OpenCV
        img_uint8 = (img * 255).astype(np.uint8)

        if self.use_otsu:
            # Use Otsu's method to automatically find optimal threshold
            _, binary = cv2.threshold(img_uint8, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # Use manual threshold
            threshold_uint8 = int(self.fallback_threshold * 255)
            binary = (img_uint8 < threshold_uint8).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return 0.0, binary, (0.0, 0.0)

        # Get largest contour (main droplet)
        largest = max(contours, key=cv2.contourArea)

        # Fit minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(largest)

        return 2 * radius, binary, (cx, cy)  # diameter, mask, center


class RealCropInference:
    """Inference on real preprocessed droplet crops."""

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = 'auto',
        inference_pixel_size_mm: Optional[float] = None,
        inference_optical_params: Optional[dict] = None,
        defocus_calibration: Optional[dict] = None,
        calibration_file: Optional[str] = None,
        inference_camera_scale_px_per_mm: Optional[float] = None,
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            config_path: Optional path to config (if not in checkpoint)
            device: 'cuda', 'cpu', or 'auto'
            inference_pixel_size_mm: Optional pixel size for converting to physical units (mm).
                                     Not used for scaling - scaling is just native_size / model_size.
            inference_optical_params: Optional dict with optical parameters for the inference camera.
                                     Keys: 'pixel_size_mm', 'focal_length_mm', 'aperture_diameter_mm',
                                     'focus_distance_mm'. Used for blur → defocus distance conversion.
            defocus_calibration: Optional dict with linear calibration parameters.
                                Keys: 'slope', 'offset'. Applies: corrected = slope * predicted + offset
            calibration_file: Optional path to calibration YAML for direct mode.
                             Must contain 'direct' section with 'rho_px_per_mm' and 'sigma_0'.
                             Only used when checkpoint training_mode == 'direct'.
                             Different from defocus_calibration (which is post-hoc correction for optical mode).
            inference_camera_scale_px_per_mm: Scale of the inference camera in px/mm. Used in direct
                                             mode only to correct rho: rho_eff = rho * (scale_inf / scale_calib).
                                             If None (or scale_calib not in config), no correction is applied.
        """
        self.inference_camera_scale_px_per_mm = inference_camera_scale_px_per_mm
        self.inference_optical_params = inference_optical_params
        self.defocus_calibration = defocus_calibration
        self.model_path = Path(model_path)

        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load checkpoint
        logger.info(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # Get config
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            raise ValueError("Config not found. Provide config_path or ensure it's in checkpoint.")

        # Detect training mode from checkpoint
        self.training_mode = checkpoint.get('training_mode', 'optical')
        self.blur_term = "σ" if self.training_mode == "direct" else "CoC"
        logger.info(f"Training mode: {self.training_mode}")

        # Load direct mode calibration if needed
        self.direct_slope = None
        self.direct_offset = None
        if self.training_mode == 'direct':
            if calibration_file is not None:
                # Load from calibration YAML file
                import yaml as _yaml
                with open(calibration_file, 'r') as f:
                    cal_data = _yaml.safe_load(f)
                if 'direct' not in cal_data:
                    raise ValueError(f"Calibration file missing 'direct' section: {calibration_file}")
                self.direct_slope = float(cal_data['direct']['rho_px_per_mm'])
                self.direct_offset = float(cal_data['direct'].get('sigma_0', 0.0))
                loo = cal_data['direct'].get('loo_cv', {})
                self.rho_std = float(loo.get('rho_std', 0.0))
                self.sigma_0_std = float(loo.get('sigma_0_std', 0.0))
                logger.info(f"  Direct calibration (from file): rho={self.direct_slope} px/mm, "
                      f"sigma_0={self.direct_offset} px  |z| = (sigma - sigma_0) / rho")
                if self.rho_std > 0:
                    logger.info(f"  LOO-CV uncertainty: rho_std={self.rho_std:.4f}, "
                          f"sigma_0_std={self.sigma_0_std:.4f}")
            else:
                # Fall back to checkpoint config (rho_direct and sigma_0 saved during training)
                training_cfg = self.config.get('training', {})
                rho = training_cfg.get('rho_direct')
                sigma_0 = training_cfg.get('sigma_0')
                if rho is None:
                    raise ValueError(
                        "Direct mode: no calibration_file provided and checkpoint config "
                        "missing 'rho_direct' in training section."
                    )
                self.direct_slope = float(rho)
                self.direct_offset = float(sigma_0) if sigma_0 is not None else 0.0
                self.rho_std = float(training_cfg.get('rho_std', 0.0))
                self.sigma_0_std = float(training_cfg.get('sigma_0_std', 0.0))
                logger.info(f"  Direct calibration (from checkpoint): rho={self.direct_slope} px/mm, "
                      f"sigma_0={self.direct_offset} px  |z| = (sigma - sigma_0) / rho")

        # Apply cross-camera scale correction to rho (direct mode only)
        # rho was measured on the calibration camera; if inference is on a different camera
        # the effective rho scales proportionally: rho_eff = rho * (scale_inf / scale_calib)
        if self.training_mode == 'direct' and self.direct_slope is not None:
            training_cfg = self.config.get('training', {})
            scale_calib = training_cfg.get('scale_calib_px_per_mm')
            if self.inference_camera_scale_px_per_mm is not None and scale_calib is not None and float(scale_calib) > 0:
                scale_factor = self.inference_camera_scale_px_per_mm / float(scale_calib)
                self.direct_slope = self.direct_slope * scale_factor
                logger.info(f"  Cross-camera scale correction applied: "
                      f"scale_inf={self.inference_camera_scale_px_per_mm:.1f} px/mm, "
                      f"scale_calib={scale_calib:.1f} px/mm, "
                      f"factor={scale_factor:.3f} → rho_eff={self.direct_slope:.6f} px/mm")
            elif self.inference_camera_scale_px_per_mm is not None and scale_calib is None:
                logger.warning("inference_camera_scale_px_per_mm provided but scale_calib_px_per_mm "
                      "not found in config — cross-camera correction skipped")

        # Create and load model
        self.model = DefocusNet.from_config(self.config).to(self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded model weights")
        elif 'dme_state_dict' in checkpoint:
            state = checkpoint['dme_state_dict']
            # Handle checkpoints saved from DMESubnet directly (no dme_subnet. prefix)
            sample_key = next(iter(state))
            if not sample_key.startswith('dme_subnet.'):
                state = {f'dme_subnet.{k}': v for k, v in state.items()}
            self.model.load_state_dict(state)
            logger.info("Loaded DME weights")

        self.model.eval()

        # Optical parameters for blur → defocus distance conversion
        # If inference_optical_params provided, create custom BlurParams for the inference camera
        if self.inference_optical_params:
            # Start with training config, override with inference params
            optics_cfg = self.config.get('optics', {}).copy()
            for key in ['pixel_size_mm', 'focal_length_mm', 'aperture_diameter_mm', 'focus_distance_mm']:
                if key in self.inference_optical_params:
                    optics_cfg[key] = self.inference_optical_params[key]

            # Get rho from blur config (camera-independent)
            blur_cfg = self.config.get('blur', {})
            rho = blur_cfg.get('rho', 0.25)

            # Calculate imaging distance from thin lens equation: 1/F = 1/u₀ + 1/d₀
            focal_length = optics_cfg.get('focal_length_mm', 70.0)
            focus_distance = optics_cfg.get('focus_distance_mm', 200.0)
            imaging_distance = optics_cfg.get('imaging_distance_mm',
                                              1.0 / (1.0/focal_length - 1.0/focus_distance) if focus_distance != focal_length else 200.0)

            self.optical_params = BlurParams(
                focal_length_mm=focal_length,
                focus_distance_mm=focus_distance,
                imaging_distance_mm=imaging_distance,
                aperture_diameter_mm=optics_cfg.get('aperture_diameter_mm', 17.5),
                pixel_size_mm=optics_cfg.get('pixel_size_mm', 0.02),
                rho=rho
            )
            logger.info(f"Using custom inference camera settings")
        else:
            self.optical_params = BlurParams.from_config(self.config)
        self.blur_calc = BlurCalculator(self.optical_params)

        # Get max blur for denormalization (at model scale)
        # Priority: checkpoint max_blur/max_coc > config > fallback
        ckpt_max = checkpoint.get('max_blur', checkpoint.get('max_coc'))
        if ckpt_max is not None:
            self.max_blur = ckpt_max
        else:
            # Fallback to config (for backwards compatibility)
            if 'training_config' in self.config:
                data_cfg = self.config['training_config'].get('data', {})
            else:
                data_cfg = self.config.get('data', {})

            if 'blur_range_px' in data_cfg:
                self.max_blur = data_cfg['blur_range_px'][1]
            elif 'defocus_range_mm' in data_cfg:
                defocus_range = tuple(data_cfg['defocus_range_mm'])
                blur_range = self.blur_calc.get_blur_range(defocus_range)
                self.max_blur = blur_range[1]
            else:
                self.max_blur = 20  # Fallback default

        # Get model size from config (what the model was trained with)
        # Model outputs blur at this scale, needs to be scaled to native resolution
        data_cfg = self.config.get('data', {})
        self.model_size = data_cfg.get('image_size_px', 128)

        # Store pixel sizes for optional conversion to physical units (mm)
        optics_cfg = self.config.get('optics', {})
        self.training_pixel_size_mm = optics_cfg.get('pixel_size_mm')
        # Priority: explicit inference_pixel_size_mm > inference_optical_params > training config
        if inference_pixel_size_mm:
            self.inference_pixel_size_mm = inference_pixel_size_mm
        elif self.inference_optical_params and 'pixel_size_mm' in self.inference_optical_params:
            self.inference_pixel_size_mm = self.inference_optical_params['pixel_size_mm']
        else:
            self.inference_pixel_size_mm = self.training_pixel_size_mm

        # Note: No cross-camera scaling needed for blur output!
        # Model outputs blur at model_scale. To get native: multiply by (native_size / model_size)
        # This works because rho is camera-independent.
        # Different cameras only matter for physical unit conversion (px → mm).

        # Diameter measurement (using Otsu's method by default)
        self.diameter_measurer = DiameterMeasurer()

        logger.info(f"Max {self.blur_term}: {self.max_blur} px (denormalization ceiling)")
        logger.info(f"Model size: {self.model_size}×{self.model_size} px")

        # Print inversion chain summary for direct mode
        if self.training_mode == 'direct' and self.direct_slope is not None:
            logger.debug(f"--- Direct Mode Inversion Chain ---")
            logger.debug(f"  rho_eff = {self.direct_slope:.6f} px/mm")
            logger.debug(f"  model_size = {self.model_size} px")
            logger.debug(f"  Formula: |z| = sigma_model × native_size / (rho_eff × model_size)")
            logger.debug(f"           |z| = sigma_model × native_size / ({self.direct_slope:.4f} × {self.model_size})")
            logger.debug(f"  native_size = actual input image dimension (detected per crop)")
            logger.debug(f"-----------------------------------")

        logger.info(f"Model ready for inference!")
        self._first_crop_logged = False  # Log full chain for first crop only

    def preprocess_image(self, img_path: Path) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        """
        Load and preprocess a single crop image.

        Images are scaled to 128x128 for model processing (fixed architecture size).
        Original image and size are preserved for output scaling.

        Returns:
            (tensor, original_img, original_size)
        """
        # Load grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        original_size = img.shape  # (H, W)

        # Scale to model size (128x128) - use INTER_AREA for downscaling, INTER_CUBIC for upscaling
        h, w = original_size
        if h > self.model_size or w > self.model_size:
            interpolation = cv2.INTER_AREA  # Better for downscaling
        else:
            interpolation = cv2.INTER_CUBIC  # Better for upscaling

        img_resized = cv2.resize(img, (self.model_size, self.model_size), interpolation=interpolation)

        # Normalize to [-1, 1]
        img_norm = (img_resized.astype(np.float32) / 255.0) * 2.0 - 1.0

        # Convert to tensor (B, C, H, W)
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(self.device)

        return img_tensor, img, original_size

    def denormalize_blur(self, blur_normalized: torch.Tensor) -> float:
        """Convert normalized blur [0, 1] (sigmoid output) to pixels."""
        return (blur_normalized * self.max_blur).item()

    def estimate_blur_from_image(self, img: np.ndarray) -> float:
        """
        Estimate blur (sigma or CoC) from a numpy image (BGR format).

        Args:
            img: BGR image as numpy array

        Returns:
            Estimated blur in pixels (scaled to original image resolution)
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        original_h, original_w = gray.shape[:2]

        # Scale to model size (128x128)
        if original_h > self.model_size or original_w > self.model_size:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        gray_resized = cv2.resize(gray, (self.model_size, self.model_size), interpolation=interpolation)

        # Normalize to [-1, 1]
        gray_norm = (gray_resized.astype(np.float32) / 255.0) * 2.0 - 1.0

        # Convert to tensor
        img_tensor = torch.from_numpy(gray_norm).unsqueeze(0).unsqueeze(0).to(self.device)

        # Run model — returns (B, 1) scalar in [0, 1]
        with torch.no_grad():
            pred_norm = self.model(img_tensor)

        # Extract blur value at model scale
        blur_px_model = self.denormalize_blur(pred_norm.squeeze())

        # Scale to original image resolution
        scale_factor = max(original_h, original_w) / self.model_size
        blur_px = blur_px_model * scale_factor

        return blur_px

    # Backward compat alias
    estimate_coc_from_image = estimate_blur_from_image

    def process_single_crop(
        self,
        img_path: Path,
        save_visualization: bool = False,
        output_dir: Optional[Path] = None,
        blur_threshold: float = 1.5
    ) -> Dict:
        """
        Process a single droplet crop.

        Args:
            blur_threshold: Blur threshold in pixels for "in focus" classification

        Returns:
            Dict with results: {
                'filename': str,
                'sigma_px' or 'coc_px': float (direct or optical mode),
                'defocus_mm': float,
                'focus_status': str ('in_focus' or 'out_of_focus'),
                'original_size': tuple,
            }
        """
        # Preprocess
        img_tensor, original_img, original_size = self.preprocess_image(img_path)

        # Run model — returns (B, 1) scalar in [0, 1]
        with torch.no_grad():
            pred_norm = self.model(img_tensor)

        # Extract blur value at model scale (128x128)
        blur_px_model = self.denormalize_blur(pred_norm.squeeze())

        # Scale blur to original image resolution
        scale_factor = max(original_size[0], original_size[1]) / self.model_size
        blur_px = blur_px_model * scale_factor

        # Classify focus status based on threshold (threshold is at original scale)
        if blur_threshold is not None and blur_px < blur_threshold:
            focus_status = 'in_focus'
        else:
            focus_status = 'out_of_focus'

        # Convert model output to defocus distance (mm)
        if self.training_mode == 'optical':
            # Optical mode: blur_px is CoC in pixels, use optical formula inverse
            defocus_mm = self.blur_calc.blur_to_defocus(blur_px, self.inference_pixel_size_mm)
            # Apply post-hoc linear correction if provided
            if self.defocus_calibration:
                slope = self.defocus_calibration.get('slope', 1.0)
                offset = self.defocus_calibration.get('offset', 0.0)
                defocus_mm = slope * defocus_mm + offset
        elif self.training_mode == 'direct':
            # Direct mode: use canonical physics module for inversion
            native_size = max(original_size[0], original_size[1])
            pred_val = pred_norm.squeeze().item()

            # Build ScalingParams — direct_slope is already cross-camera corrected,
            # so set s_inference == s_calib to avoid double-correction
            sigma_0_raw = self.direct_offset if self.direct_offset is not None else 0.0
            training_cfg = self.config.get('training', {})
            scale_calib = training_cfg.get('scale_calib_px_per_mm')
            s_inf = self.inference_camera_scale_px_per_mm or 1.0
            s_cal = float(scale_calib) if scale_calib and float(scale_calib) > 0 else s_inf

            # Use raw (uncorrected) rho — physics module applies cross-camera itself
            rho_raw = self.direct_slope / (s_inf / s_cal) if s_cal > 0 else self.direct_slope

            params = ScalingParams(
                rho=rho_raw, sigma_0=sigma_0_raw,
                s_calib=s_cal, s_inference=s_inf,
                max_blur=self.max_blur, model_size=self.model_size,
            )
            inv = invert_prediction(pred_val, params, native_size)
            defocus_mm = inv.defocus_mm

            # Log full inversion chain for the first crop
            if not self._first_crop_logged:
                self._first_crop_logged = True
                logger.debug(f"--- First crop inversion trace ({img_path.name}) ---")
                logger.debug(f"  Input size: {original_size[1]}x{original_size[0]} -> native_size = {native_size}")
                logger.debug(f"  pred_norm={inv.pred_norm:.4f}, sigma_model={inv.sigma_model:.4f} px")
                logger.debug(f"  sigma_native={inv.sigma_native:.4f} px, defocus={inv.defocus_mm:.4f} mm")
                logger.debug(f"  rho_eff={params.rho_eff:.4f}, sigma_0_eff={params.sigma_0_eff:.4f}")
                logger.debug(f"  clamped={inv.clamped}, saturated={inv.saturated}")
                logger.debug(f"-----------------------------------------------")
        else:
            raise ValueError(f"Unknown training_mode: {self.training_mode}")

        # Compute calibration uncertainty if available
        unc_mm = 0.0
        rho_std = getattr(self, 'rho_std', 0.0)
        sigma_0_std = getattr(self, 'sigma_0_std', 0.0)
        if rho_std > 0 and self.training_mode == 'direct' and self.direct_slope > 0:
            unc_mm = defocus_uncertainty(
                blur_px, self.direct_slope,
                self.direct_offset if self.direct_offset else 0.0,
                rho_std, sigma_0_std,
            )

        # Measure diameter from original image
        diameter_original, _, _ = self.diameter_measurer.measure_diameter(original_img / 255.0)

        # Use correct column name: sigma_px for direct mode, coc_px for optical mode
        blur_col = 'sigma_px' if self.training_mode == 'direct' else 'coc_px'
        results = {
            'filename': img_path.name,
            blur_col: blur_px,
            'defocus_mm': defocus_mm,
            'defocus_uncertainty_mm': unc_mm,
            'focus_status': focus_status,
            'diameter_original_px': diameter_original,
            'original_size': original_size
        }

        # Save visualization
        if save_visualization and output_dir is not None:
            viz_dir = output_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True, parents=True)

            self._create_comparison_viz(
                original_img,
                blur_px,
                defocus_mm,
                diameter_original,
                img_path.stem,
                viz_dir,
                defocus_uncertainty_mm=unc_mm,
            )

        return results

    def _create_comparison_viz(
        self,
        original: np.ndarray,
        blur_px: float,
        defocus_mm: float,
        diameter_original: float,
        name: str,
        output_dir: Path,
        defocus_uncertainty_mm: float = 0.0,
    ):
        """Create visualization showing the original image with blur/defocus annotations."""

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

        unc_str = f" \u00b1 {defocus_uncertainty_mm:.2f}" if defocus_uncertainty_mm > 0 else ""
        ax.imshow(original, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'{name}\n{self.blur_term}: {blur_px:.2f} px | Defocus: {defocus_mm:.2f}{unc_str} mm',
                     fontsize=12, fontweight='bold')
        ax.axis('off')
        self._add_diameter_arrow(ax, original / 255.0)

        plt.suptitle(
            f'Diameter: {diameter_original:.1f} px',
            fontsize=11, fontweight='bold', y=1.02
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_dir / f'{name}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _add_diameter_arrow(self, ax, img: np.ndarray):
        """Add a horizontal arrow showing diameter measurement.

        Recomputes diameter independently to ensure perfect alignment.
        """
        # Measure diameter with independent Otsu computation
        diameter, binary, (cx, cy) = self.diameter_measurer.measure_diameter(img)

        if diameter == 0:
            return

        radius = diameter / 2.0

        # Draw horizontal arrow across diameter
        y_pos = cy
        x_start = cx - radius
        x_end = cx + radius

        # Draw double-headed arrow
        ax.annotate('', xy=(x_end, y_pos), xytext=(x_start, y_pos),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))

        # Add text label
        ax.text(cx, y_pos - radius * 0.3, f'{diameter:.1f} px',
               color='red', fontsize=10, fontweight='bold',
               ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))

    def process_material_folder(
        self,
        material_dir: Path,
        output_base: Path,
        save_visualizations: bool = True,
        viz_sample_rate: int = 10,
        blur_threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Process all crops in a single material folder.

        Args:
            material_dir: Path to material folder (e.g., OUTPUT/8mm-borosilicate-2)
            output_base: Base output directory
            save_visualizations: Save comparison visualizations
            viz_sample_rate: Save visualization for every Nth crop (to avoid thousands of images)
            blur_threshold: Blur threshold in pixels for "in focus" classification

        Returns:
            DataFrame with results
        """
        material_name = material_dir.name
        logger.info(f"{'='*60}")
        logger.info(f"Processing: {material_name}")
        logger.info(f"{'='*60}")

        # Find all crop files - first try *_crop.png, then fall back to all *.png
        crop_files = sorted(list(material_dir.glob('*_crop.png')))

        if len(crop_files) == 0:
            # Fall back to all PNG files if no *_crop.png found
            crop_files = sorted(list(material_dir.glob('*.png')))
            if len(crop_files) > 0:
                logger.info(f"No *_crop.png files found, using all {len(crop_files)} PNG files")

        if len(crop_files) == 0:
            logger.warning(f"No image files found in {material_dir}")
            return pd.DataFrame()

        logger.info(f"Found {len(crop_files)} image files")

        # Create output directory for this material
        output_dir = output_base / material_name
        output_dir.mkdir(exist_ok=True, parents=True)

        # Process each crop
        results = []

        for i, crop_path in enumerate(tqdm(crop_files, desc=f"Processing {material_name}")):
            # Decide whether to save visualization (sample every Nth)
            save_viz = save_visualizations and (i % viz_sample_rate == 0)

            try:
                result = self.process_single_crop(
                    crop_path,
                    save_visualization=save_viz,
                    output_dir=output_dir,
                    blur_threshold=blur_threshold
                )
                result['material'] = material_name
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing {crop_path.name}: {e}")
                continue

        # Create DataFrame
        df = pd.DataFrame(results)

        if len(df) > 0:
            # Save CSV
            csv_path = output_dir / 'blur_results.csv'
            df.to_csv(csv_path, index=False)

            # Print statistics — use correct column name for mode
            blur_col = 'sigma_px' if self.training_mode == 'direct' else 'coc_px'
            logger.info(f"{material_name} Statistics:")
            logger.info(f"  Processed: {len(df)} crops")
            logger.info(f"  {self.blur_term} Mean:  {df[blur_col].mean():.2f} px")
            logger.info(f"  {self.blur_term} Std:   {df[blur_col].std():.2f} px")
            logger.info(f"  {self.blur_term} Range: {df[blur_col].min():.2f} - {df[blur_col].max():.2f} px")
            logger.info(f"  Defocus Mean: {df['defocus_mm'].mean():.2f} mm")
            logger.info(f"  Diameter (original): {df['diameter_original_px'].mean():.2f} ± {df['diameter_original_px'].std():.2f} px")
            logger.info(f"Saved results to: {csv_path}")

        return df

    def process_all_materials(
        self,
        input_base: Path,
        output_base: Path,
        save_visualizations: bool = True,
        viz_sample_rate: int = 10
    ):
        """
        Process all material folders in the input directory.

        Args:
            input_base: Base input directory (e.g., OUTPUT/)
            output_base: Base output directory
            save_visualizations: Save comparison visualizations
            viz_sample_rate: Save 1 visualization per N crops
        """
        input_base = Path(input_base)
        output_base = Path(output_base)
        output_base.mkdir(exist_ok=True, parents=True)

        logger.info(f"{'='*60}")
        logger.info(f"REAL DROPLET CROP INFERENCE")
        logger.info(f"{'='*60}")
        logger.info(f"Input:  {input_base}")
        logger.info(f"Output: {output_base}")

        # Find all material subdirectories OR process flat folder
        material_dirs = [d for d in input_base.iterdir() if d.is_dir()]
        all_results = []

        if len(material_dirs) > 0:
            # Standard mode: process subfolders as materials
            logger.info(f"Found {len(material_dirs)} material folders:")
            for d in material_dirs:
                crop_files = list(d.glob('*_crop.png'))
                if len(crop_files) == 0:
                    crop_files = list(d.glob('*.png'))
                logger.info(f"  - {d.name}: {len(crop_files)} images")

            # Process each material
            for material_dir in material_dirs:
                df = self.process_material_folder(
                    material_dir,
                    output_base,
                    save_visualizations=save_visualizations,
                    viz_sample_rate=viz_sample_rate
                )

                if len(df) > 0:
                    all_results.append(df)
        else:
            # Flat folder mode: process input_base directly as single "material"
            direct_crops = list(input_base.glob('*_crop.png'))
            if len(direct_crops) == 0:
                direct_crops = list(input_base.glob('*.png'))

            if len(direct_crops) == 0:
                logger.warning(f"No images found in {input_base}")
                return

            logger.info(f"No subfolders found - processing {len(direct_crops)} images directly")

            df = self.process_material_folder(
                input_base,
                output_base,
                save_visualizations=save_visualizations,
                viz_sample_rate=viz_sample_rate
            )

            if len(df) > 0:
                all_results.append(df)

        # Combine all results
        if len(all_results) > 0:
            combined_df = pd.concat(all_results, ignore_index=True)

            # Save combined results
            summary_path = output_base / 'summary_all_materials.csv'
            combined_df.to_csv(summary_path, index=False)
            logger.info(f"Saved combined summary to: {summary_path}")

            # Create summary visualizations
            self._create_summary_plots(combined_df, output_base)

            # Print overall statistics
            logger.info(f"{'='*60}")
            logger.info(f"OVERALL STATISTICS")
            logger.info(f"{'='*60}")
            logger.info(f"Total crops processed: {len(combined_df)}")
            logger.info(f"Materials: {combined_df['material'].nunique()}")
            blur_col = 'sigma_px' if self.training_mode == 'direct' else 'coc_px'
            logger.info(f"{self.blur_term} Statistics (all materials):")
            logger.info(f"  Mean:   {combined_df[blur_col].mean():.2f} px")
            logger.info(f"  Median: {combined_df[blur_col].median():.2f} px")
            logger.info(f"  Std:    {combined_df[blur_col].std():.2f} px")
            logger.info(f"  Range:  {combined_df[blur_col].min():.2f} - {combined_df[blur_col].max():.2f} px")


            logger.info(f"Defocus Statistics (all materials):")
            logger.info(f"  Mean:   {combined_df['defocus_mm'].mean():.2f} mm")
            logger.info(f"  Median: {combined_df['defocus_mm'].median():.2f} mm")
            logger.info(f"  Range:  {combined_df['defocus_mm'].min():.2f} - {combined_df['defocus_mm'].max():.2f} mm")

            logger.info(f"{'='*60}")
            logger.info(f"INFERENCE COMPLETE!")
            logger.info(f"{'='*60}")

    @staticmethod
    def _parse_true_z(filename: str) -> Optional[float]:
        """Extract true z from filename pattern like '7_mm_10_z-6.20mm.png'."""
        match = re.search(r'z([+-]?\d+\.?\d*)mm', filename)
        if match:
            return float(match.group(1))
        return None

    def _create_summary_plots(self, df: pd.DataFrame, output_dir: Path):
        """Create summary visualizations.

        If true z-positions can be parsed from filenames, produces a 5-panel
        z-stack validation figure (pred vs true, residuals, symmetry, per-range
        MAE, sample strip). Otherwise falls back to basic distribution plots.
        """
        blur_col = 'sigma_px' if 'sigma_px' in df.columns else 'coc_px'

        # Try to parse true z from filenames
        df = df.copy()
        df['true_z'] = df['filename'].apply(self._parse_true_z)
        has_ground_truth = df['true_z'].notna().sum() > 0

        if has_ground_truth:
            self._create_zstack_validation_plots(df, blur_col, output_dir)
        else:
            self._create_basic_summary_plots(df, blur_col, output_dir)

    def _create_zstack_validation_plots(self, df: pd.DataFrame, blur_col: str, output_dir: Path):
        """Create 5-panel z-stack validation figure when ground truth is available."""
        import numpy as np
        from scipy import stats

        df = df.copy()
        df['true_defocus'] = df['true_z'].abs()
        df['pred_defocus'] = df['defocus_mm']
        df['residual'] = df['pred_defocus'] - df['true_defocus']
        df = df.sort_values('true_z').reset_index(drop=True)

        # Keep full df for sample strip, filter for analysis
        df_all = df.copy()

        # Exclude frames outside calibration range:
        #   near-focus (|z| < 0.5 mm) and plateau (|z| > 6.8 mm)
        n_before = len(df)
        df = df[(df['true_defocus'] >= 0.5) & (df['true_defocus'] <= 6.8)].reset_index(drop=True)
        n_excluded = n_before - len(df)
        if n_excluded > 0:
            logger.info(f"  Excluded {n_excluded} frames outside calibration range "
                  f"(|z| < 0.5 mm or |z| > 6.8 mm), {len(df)} remaining")

        # --- Figure 1: Main 4-panel analysis ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Predicted vs True Defocus (the money plot)
        ax = axes[0, 0]
        neg_mask = df['true_z'] < 0
        pos_mask = df['true_z'] >= 0
        ax.scatter(df.loc[neg_mask, 'true_defocus'], df.loc[neg_mask, 'pred_defocus'],
                   s=40, alpha=0.8, color='#3B6CB5', label='−z (below focus)', zorder=3)
        ax.scatter(df.loc[pos_mask, 'true_defocus'], df.loc[pos_mask, 'pred_defocus'],
                   s=40, alpha=0.8, color='#C0544E', label='+z (above focus)', zorder=3)
        # Perfect line
        lim_max = max(df['true_defocus'].max(), df['pred_defocus'].max()) * 1.05
        ax.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.4, linewidth=1, label='Perfect')
        # Linear fit
        slope, intercept, r_value, _, _ = stats.linregress(df['true_defocus'], df['pred_defocus'])
        fit_x = np.linspace(0, lim_max, 100)
        ax.plot(fit_x, slope * fit_x + intercept, 'r-', linewidth=1.5,
                label=f'Fit: y = {slope:.3f}x + {intercept:.2f}')
        ax.set_xlabel('True |z| (mm)', fontsize=11)
        ax.set_ylabel('Predicted |z| (mm)', fontsize=11)
        ax.set_title(f'Predicted vs True Defocus (R² = {r_value**2:.3f}, n={len(df)})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_aspect('equal')

        # Panel 2: Signed Residual vs True z (keeps sign)
        ax = axes[0, 1]
        ax.scatter(df['true_z'], df['residual'], s=40, alpha=0.7, color='#3B6CB5', zorder=3)
        ax.axhline(0, color='k', linestyle='--', alpha=0.4, linewidth=1)
        ax.set_xlabel('True z (mm, signed)', fontsize=11)
        ax.set_ylabel('Residual: pred − true |z| (mm)', fontsize=11)
        ax.set_title('Signed Residual vs True Position', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Annotate mean bias and MAE
        mae = df['residual'].abs().mean()
        bias = df['residual'].mean()
        ax.text(0.02, 0.98, f'MAE = {mae:.3f} mm\nBias = {bias:+.3f} mm',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Panel 3: Symmetry plot (+z vs -z at matched |z|)
        ax = axes[1, 0]
        # Pair +z and -z predictions at same |z|
        neg_df = df[df['true_z'] < 0][['true_defocus', 'pred_defocus']].rename(
            columns={'pred_defocus': 'pred_neg'})
        pos_df = df[df['true_z'] > 0][['true_defocus', 'pred_defocus']].rename(
            columns={'pred_defocus': 'pred_pos'})
        sym = pd.merge(neg_df, pos_df, on='true_defocus', how='inner')

        if len(sym) > 0:
            ax.scatter(sym['pred_neg'], sym['pred_pos'], s=50, alpha=0.8, color='#4A9E4A', zorder=3)
            sym_lim = max(sym['pred_neg'].max(), sym['pred_pos'].max()) * 1.05
            ax.plot([0, sym_lim], [0, sym_lim], 'k--', alpha=0.4, linewidth=1, label='Perfect symmetry')
            ax.set_xlim(0, sym_lim)
            ax.set_ylim(0, sym_lim)
            ax.set_aspect('equal')
            max_diff = (sym['pred_neg'] - sym['pred_pos']).abs().max()
            mean_diff = (sym['pred_neg'] - sym['pred_pos']).abs().mean()
            ax.text(0.02, 0.98, f'Max |diff| = {max_diff:.3f} mm\nMean |diff| = {mean_diff:.3f} mm',
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            ax.legend(fontsize=9)
        ax.set_xlabel('Predicted |z| at −z (mm)', fontsize=11)
        ax.set_ylabel('Predicted |z| at +z (mm)', fontsize=11)
        ax.set_title('Symmetry: +z vs −z Predictions', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Panel 4: Per-range MAE bar chart (1mm bins)
        ax = axes[1, 1]
        df['z_bin'] = pd.cut(df['true_defocus'], bins=np.arange(0, df['true_defocus'].max() + 1.5, 1.0))
        bin_stats = df.groupby('z_bin', observed=True).agg(
            mae=('residual', lambda x: x.abs().mean()),
            n=('residual', 'count'),
            bias=('residual', 'mean')
        ).reset_index()
        bin_labels = [f'{int(b.left)}–{int(b.right)}' for b in bin_stats['z_bin']]
        colours = ['#3B6CB5', '#4A9E4A', '#E8833A', '#C0544E', '#7B5EA7',
                   '#C8A835', '#5C5C5C', '#2ecc71', '#e74c3c']
        bar_colours = [colours[i % len(colours)] for i in range(len(bin_stats))]
        bars = ax.bar(range(len(bin_stats)), bin_stats['mae'], color=bar_colours,
                      edgecolor='white', linewidth=1.5)
        ax.set_xticks(range(len(bin_stats)))
        ax.set_xticklabels(bin_labels)
        ax.set_xlabel('True |z| range (mm)', fontsize=11)
        ax.set_ylabel('MAE (mm)', fontsize=11)
        ax.set_title('Per-Range MAE', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, mae_val, n in zip(bars, bin_stats['mae'], bin_stats['n']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mae_val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'n={n}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        if len(bin_stats) > 0 and bin_stats['mae'].max() > 0:
            ax.set_ylim(top=bin_stats['mae'].max() * 1.25)

        plt.tight_layout()
        plot_path = output_dir / 'summary_analysis.png'
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved z-stack validation plots to: {plot_path}")

        # --- Figure 2: Sample crop strip (use full df including excluded frames) ---
        self._create_sample_strip(df_all, output_dir)

        # --- Print validation summary to console and file ---
        self._print_validation_summary(df, slope, intercept, r_value, output_dir)

    def _create_sample_strip(self, df: pd.DataFrame, output_dir: Path):
        """Create a strip of sample crops at evenly-spaced z-positions."""
        import numpy as np

        # Pick ~8 evenly-spaced z-positions across the full range
        z_values = df['true_z'].sort_values().unique()
        n_samples = min(8, len(z_values))
        indices = np.linspace(0, len(z_values) - 1, n_samples, dtype=int)
        selected_z = z_values[indices]

        # Find the parent directory of crops (look for the image files)
        sample_rows = []
        for z in selected_z:
            row = df[df['true_z'] == z].iloc[0]
            sample_rows.append(row)

        if len(sample_rows) == 0:
            return

        fig, axes = plt.subplots(1, len(sample_rows), figsize=(3 * len(sample_rows), 4))
        if len(sample_rows) == 1:
            axes = [axes]

        for ax, row in zip(axes, sample_rows):
            # Try to load the image from the material folder
            # Reconstruct path: output_dir / material / visualizations or search for file
            img = None
            for search_dir in output_dir.rglob('*.png'):
                if row['filename'].replace('.png', '') in search_dir.name:
                    img = cv2.imread(str(search_dir), cv2.IMREAD_GRAYSCALE)
                    break

            if img is None:
                # Try loading from the original input path
                for parent in output_dir.parent.rglob(row['filename']):
                    img = cv2.imread(str(parent), cv2.IMREAD_GRAYSCALE)
                    break

            if img is not None:
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            else:
                ax.text(0.5, 0.5, 'Image\nnot found', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9)

            ax.set_title(f'z = {row["true_z"]:+.1f} mm', fontsize=10, fontweight='bold')
            pred = row['defocus_mm']
            true = abs(row['true_z'])
            ax.set_xlabel(f'Pred: {pred:.2f} mm\nTrue: {true:.1f} mm', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle('Sample Crops Across Z-Stack', fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        strip_path = output_dir / 'sample_strip.png'
        plt.savefig(strip_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved sample strip to: {strip_path}")

    def _print_validation_summary(self, df: pd.DataFrame, slope: float,
                                   intercept: float, r_value: float, output_dir: Path):
        """Print and save a text summary of z-stack validation results."""
        import numpy as np

        true_def = df['true_defocus']
        pred_def = df['pred_defocus']
        residual = df['residual']

        lines = [
            '=' * 60,
            'Z-STACK VALIDATION SUMMARY',
            '(filtered to calibration range: 0.5 <= |z| <= 6.8 mm)',
            '=' * 60,
            f'Samples: {len(df)}',
            f'True z range: {df["true_z"].min():.1f} to {df["true_z"].max():.1f} mm',
            f'True |z| range: {true_def.min():.1f} to {true_def.max():.1f} mm',
            f'Pred |z| range: {pred_def.min():.2f} to {pred_def.max():.2f} mm',
            '',
            '--- Accuracy ---',
            f'R² = {r_value**2:.4f}',
            f'Linear fit: pred = {slope:.4f} × true + {intercept:.4f}',
            f'MAE = {residual.abs().mean():.3f} mm',
            f'RMSE = {np.sqrt((residual**2).mean()):.3f} mm',
            f'Median AE = {residual.abs().median():.3f} mm',
            f'Bias = {residual.mean():+.3f} mm',
            f'Max error = {residual.abs().max():.3f} mm',
            '',
            '--- Per-Range ---',
        ]
        for lo in np.arange(0, true_def.max() + 0.5, 1.0):
            hi = lo + 1.0
            mask = (true_def >= lo) & (true_def < hi)
            n = mask.sum()
            if n > 0:
                mae = residual[mask].abs().mean()
                bias = residual[mask].mean()
                lines.append(f'  |z| {lo:.0f}–{hi:.0f} mm: n={n:>3d}, MAE={mae:.3f}, bias={bias:+.3f}')

        # Symmetry
        neg = df[df['true_z'] < 0][['true_defocus', 'pred_defocus']].rename(
            columns={'pred_defocus': 'pred_neg'})
        pos = df[df['true_z'] > 0][['true_defocus', 'pred_defocus']].rename(
            columns={'pred_defocus': 'pred_pos'})
        sym = pd.merge(neg, pos, on='true_defocus', how='inner')
        if len(sym) > 0:
            lines.append('')
            lines.append('--- Symmetry (matched |z|) ---')
            lines.append(f'Paired points: {len(sym)}')
            lines.append(f'Max |diff|: {(sym["pred_neg"] - sym["pred_pos"]).abs().max():.3f} mm')
            lines.append(f'Mean |diff|: {(sym["pred_neg"] - sym["pred_pos"]).abs().mean():.3f} mm')

        lines.append('')
        lines.append('=' * 60)

        summary_text = '\n'.join(lines)
        logger.info(summary_text)

        txt_path = output_dir / 'validation_summary.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        logger.info(f"Saved validation summary to: {txt_path}")

    def _create_basic_summary_plots(self, df: pd.DataFrame, blur_col: str, output_dir: Path):
        """Fallback summary when no ground truth z is available in filenames."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. Blur distribution
        materials = df['material'].unique()
        for material in materials:
            data = df[df['material'] == material][blur_col]
            axes[0].hist(data, bins=30, alpha=0.6, label=material, edgecolor='black')
        axes[0].set_xlabel(f'{self.blur_term} (px)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title(f'{self.blur_term} Distribution', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        # 2. Defocus distribution
        axes[1].hist(df['defocus_mm'], bins=40, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1].set_xlabel('Defocus Distance (mm)', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Defocus Distribution', fontsize=12, fontweight='bold')
        axes[1].axvline(df['defocus_mm'].mean(), color='r', linestyle='--',
                        label=f"Mean: {df['defocus_mm'].mean():.2f} mm")
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)

        # 3. Blur boxplot by material
        df.boxplot(column=blur_col, by='material', ax=axes[2])
        axes[2].set_xlabel('Material', fontsize=11)
        axes[2].set_ylabel(f'{self.blur_term} (px)', fontsize=11)
        axes[2].set_title(f'{self.blur_term} by Material', fontsize=12, fontweight='bold')
        axes[2].get_figure().suptitle('')
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

        plt.tight_layout()
        plot_path = output_dir / 'summary_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved summary plots to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on real preprocessed droplet crops"
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='training_output/checkpoints/best_model.pth',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to config YAML (optional if in checkpoint)'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='',
        help='Input directory containing material subfolders with crops'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='inference_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip creating visualization comparisons'
    )
    parser.add_argument(
        '--viz-rate',
        type=int,
        default=10,
        help='Save 1 visualization per N crops (default: 10)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device: cuda, cpu, or auto (default: auto)'
    )

    args = parser.parse_args()

    # Create inference engine
    inference = RealCropInference(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )

    # Process all materials
    inference.process_all_materials(
        input_base=args.input,
        output_base=args.output,
        save_visualizations=not args.no_viz,
        viz_sample_rate=args.viz_rate
    )


if __name__ == "__main__":
    main()
