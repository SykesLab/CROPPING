"""
Synthetic Blur Generator

Generates training data by applying physics-based Gaussian blur to sharp
droplet images, simulating the defocus blur in shadowgraphy systems.

In optical mode, the blur is determined by the circle of confusion (CoC) formula:

    coc = D_lens × u₀ × |1/F - 1/u₀ - 1/(d + d₀)|

where:
    - coc: Circle of confusion diameter (pixels)
    - D_lens: Aperture diameter (mm)
    - u₀: Imaging distance (mm)
    - F: Focal length (mm)
    - d₀: Focus distance (mm)
    - d: Out-of-focus distance (mm) — what we want to estimate

Reference:
    Wang, Z. et al. (2022). Physics of Fluids, 34(7), 073301.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass
import yaml
from tqdm import tqdm
import random
import argparse


@dataclass
class BlurParams:
    """Optical system parameters."""
    focal_length_mm: float      # F
    focus_distance_mm: float    # d₀
    imaging_distance_mm: float  # u₀
    aperture_diameter_mm: float  # D_lens
    pixel_size_mm: float        # Physical pixel size
    rho: float = 1.0            # Gaussian blur constant
    # Cross-resolution/camera scaling parameters
    calib_pixel_size_mm: Optional[float] = None      # Calibration camera pixel size
    calib_reference_resolution: Optional[int] = None  # Calibration image resolution
    training_crop_size_px: Optional[int] = None       # Training crop size before resize
    training_image_size_px: int = 128                 # Training image size after resize
    # Cross-camera scaling
    # px/mm of calibration camera (from calibration GUI)
    scale_calib_px_per_mm: Optional[float] = None
    # Dual-mode training support
    training_mode: str = "optical"                    # "optical" or "direct"
    # Direct mode: px/mm (only used if training_mode == "direct")
    rho_direct: Optional[float] = None
    # Loaded from config for backwards compat; NOT used in direct mode training (crops carry native blur)
    sigma_0: Optional[float] = None

    @property
    def f_number(self) -> float:
        """Calculate f-number from focal length and aperture diameter."""
        return self.focal_length_mm / self.aperture_diameter_mm

    def describe(self) -> str:
        """Return human-readable description of optical parameters."""
        return (
            f"Optical Parameters:\n"
            f"  Focal length: {self.focal_length_mm} mm\n"
            f"  Aperture diameter: {self.aperture_diameter_mm} mm (f/{self.f_number:.1f})\n"
            f"  Focus distance: {self.focus_distance_mm} mm\n"
            f"  Imaging distance: {self.imaging_distance_mm} mm\n"
            f"  Pixel size: {self.pixel_size_mm} mm\n"
            f"  rho: {self.rho}"
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BlurParams':
        """
        Load from configuration dictionary.

        Accepts both training format (aperture_diameter_mm) and
        calibration format (f_number). If both are present, aperture_diameter_mm
        takes precedence. If only f_number is provided, aperture diameter is
        calculated as focal_length / f_number.

        Also checks for 'training_config' section from calibration exports.
        """
        # Check for training_config section (from calibration export)
        if 'training_config' in config:
            config = config['training_config']

        optics = config.get('optics', config.get('optical_params', {}))
        blur = config.get('blur', {})

        focal_length_mm = optics.get('focal_length_mm', 200.0)

        # Handle aperture: accept either aperture_diameter_mm or f_number
        if 'aperture_diameter_mm' in optics:
            aperture_diameter_mm = optics['aperture_diameter_mm']
        elif 'f_number' in optics:
            # Convert f-number to aperture diameter: D = f / N
            f_number = optics['f_number']
            aperture_diameter_mm = focal_length_mm / f_number
        else:
            aperture_diameter_mm = 50.0  # Default

        # Handle imaging_distance: calculate from thin lens equation if not provided
        focus_distance_mm = optics.get('focus_distance_mm', 400.0)
        if 'imaging_distance_mm' in optics:
            imaging_distance_mm = optics['imaging_distance_mm']
        else:
            # Calculate using thin lens equation: 1/f = 1/D + 1/u0
            # u0 = f × D / (D - f)
            if focus_distance_mm > focal_length_mm:
                imaging_distance_mm = focal_length_mm * \
                    focus_distance_mm / (focus_distance_mm - focal_length_mm)
            else:
                imaging_distance_mm = focus_distance_mm  # Fallback

        # Handle rho: check multiple possible locations
        rho = blur.get('rho', config.get('formula_rho', 1.0))

        # Get calibration reference parameters (for cross-resolution/camera scaling)
        calib_pixel_size = None
        calib_reference_resolution = None
        calibration = config.get('calibration', {})
        if calibration:
            calib_pixel_size = calibration.get('pixel_size_mm')
            calib_reference_resolution = calibration.get('reference_resolution')

        # Get training parameters
        training = config.get('training', {})
        training_crop_size = training.get('crop_size_px')
        training_image_size = config.get('data', {}).get('image_size_px', 128)

        # Get dual-mode parameters
        training_mode = training.get('training_mode', 'optical')
        rho_direct = training.get('rho_direct')
        sigma_0 = training.get('sigma_0')
        scale_calib = training.get('scale_calib_px_per_mm')

        # Validate direct mode parameters
        if training_mode == "direct" and rho_direct is None:
            raise ValueError("Direct mode requires 'rho_direct' parameter in training config")

        return cls(
            focal_length_mm=focal_length_mm,
            focus_distance_mm=focus_distance_mm,
            imaging_distance_mm=imaging_distance_mm,
            aperture_diameter_mm=aperture_diameter_mm,
            pixel_size_mm=optics.get('pixel_size_mm', 0.02),
            rho=rho,
            calib_pixel_size_mm=calib_pixel_size,
            calib_reference_resolution=calib_reference_resolution,
            training_crop_size_px=training_crop_size,
            training_image_size_px=training_image_size,
            training_mode=training_mode,
            rho_direct=rho_direct,
            sigma_0=sigma_0,
            scale_calib_px_per_mm=scale_calib,
        )


class BlurCalculator:
    """
    Blur calculator for defocus estimation.

    Implements the relationship between physical defocus distance (Δz)
    and blur kernel size. Supports both optical (CoC) and direct (sigma) modes.
    """

    def __init__(self, params: BlurParams, calibration_model=None):
        """
        Args:
            params: Optical system parameters
            calibration_model: Optional ``physics.CalibrationModel``. When
                provided, ``defocus_to_sigma`` (direct mode) delegates to
                its ``forward`` method — supports linear / quadrature /
                hybrid. When None, the legacy linear formula via
                ``params.rho_direct`` / ``params.sigma_0`` is used.
        """
        self.params = params
        self.calibration_model = calibration_model

        # Pre-compute constants
        self.F = params.focal_length_mm
        self.d0 = params.focus_distance_mm
        self.u0 = params.imaging_distance_mm
        self.D_lens = params.aperture_diameter_mm
        self.pixel_size = params.pixel_size_mm
        self.rho = params.rho

    def defocus_to_coc_mm(self, d: float) -> float:
        """
        Calculate CoC diameter in mm from defocus distance.

        Args:
            d: Out-of-focus distance in mm (positive or negative)

        Returns:
            Circle of confusion diameter in mm
        """
        # CoC formula from Wang et al. Eq. (2)
        term1 = 1.0 / self.F - 1.0 / self.u0
        denominator = d + self.d0
        if abs(denominator) < 1e-10:
            return 0.0
        term2 = 1.0 / denominator
        coc = self.D_lens * self.u0 * abs(term1 - term2)
        return coc

    def defocus_to_coc_px(self, d: float) -> float:
        """
        Calculate CoC diameter in pixels from defocus distance.

        Args:
            d: Out-of-focus distance in mm

        Returns:
            Circle of confusion diameter in pixels
        """
        coc_mm = self.defocus_to_coc_mm(d)
        return coc_mm / self.pixel_size

    def coc_to_sigma(self, coc_px: float) -> float:
        """
        Convert CoC to Gaussian blur sigma.

        Simple relationship: σ = ρ × coc

        Note: coc_px should already be at model scale (scaled by model_size/crop_size
        during training data generation). No additional scaling needed here.

        Args:
            coc_px: Circle of confusion in pixels (at model scale)

        Returns:
            Gaussian kernel standard deviation
        """
        return self.rho * coc_px

    def defocus_to_sigma(self, defocus_mm: float) -> float:
        """
        Convert defocus distance to Gaussian blur sigma.

        Supports two modes:
        - "optical": σ = ρ × CoC(defocus) [Wang et al. 2022 formula]
        - "direct": σ = ρ_direct × |defocus| [linear calibration, no σ₀ offset]

        Args:
            defocus_mm: Defocus distance in mm

        Returns:
            Gaussian kernel standard deviation (pixels)
        """
        mode = getattr(self.params, 'training_mode', 'optical')

        if mode == "optical":
            # Existing behavior: defocus → CoC → sigma
            coc_px = self.defocus_to_coc_px(defocus_mm)
            return self.coc_to_sigma(coc_px)
        elif mode == "direct":
            # Direct mode: σ(z) at calibration native pixel scale.
            #
            # If a CalibrationModel was provided (Phase 5), delegate so the
            # selected method (linear/quadrature/hybrid) is honoured.
            # Otherwise fall back to the legacy linear formula via
            # rho_direct / sigma_0 — byte-identical to pre-Phase-5
            # behaviour for old callers.
            cm = getattr(self, 'calibration_model', None)
            if cm is not None:
                return float(cm.forward(abs(defocus_mm)))
            sigma_0 = self.params.sigma_0 if self.params.sigma_0 is not None else 0.0
            return self.params.rho_direct * abs(defocus_mm) + sigma_0
        else:
            raise ValueError(f"Unknown training_mode: {mode}. Must be 'optical' or 'direct'")

    def blur_to_defocus(self, coc_px: float, pixel_size_mm: float = None) -> float:
        """
        Inverse: estimate defocus distance from blur value.

        Note: Returns absolute value (sign ambiguity).

        Args:
            coc_px: Blur value in pixels (CoC or sigma, at native resolution)
            pixel_size_mm: Pixel size to use for conversion. If None, uses
                          the pixel size from optical params (training camera).
                          For cross-camera inference, pass the inference camera's
                          pixel size.

        Returns:
            Estimated absolute defocus distance in mm
        """
        # Direct mode: inverse of σ = ρ_direct × |d|
        mode = getattr(self.params, 'training_mode', 'optical')
        if mode == "direct":
            rho = self.params.rho_direct
            return max(0.0, coc_px / rho)

        # Optical mode below
        if pixel_size_mm is None:
            pixel_size_mm = self.pixel_size

        coc_mm = coc_px * pixel_size_mm

        # Solve for d from: coc = D_lens × u₀ × |1/F - 1/u₀ - 1/(d + d₀)|
        # |1/F - 1/u₀ - 1/(d + d₀)| = coc / (D_lens × u₀)

        if coc_mm == 0:
            return 0.0

        rhs = coc_mm / (self.D_lens * self.u0)
        base = 1.0 / self.F - 1.0 / self.u0

        # Two possible solutions (in front / behind focal plane)
        # 1/(d + d₀) = base - rhs  or  1/(d + d₀) = base + rhs

        try:
            d1 = 1.0 / (base - rhs) - self.d0
            d2 = 1.0 / (base + rhs) - self.d0

            # Return the one with smaller absolute value
            # (closer to focal plane is more likely)
            return min(abs(d1), abs(d2))
        except ZeroDivisionError:
            return float('inf')

    def get_coc_range(self, defocus_range: Tuple[float, float]) -> Tuple[float, float]:
        """
        Get CoC range for a given defocus range.

        Args:
            defocus_range: (min_d, max_d) in mm

        Returns:
            (min_coc, max_coc) in pixels
        """
        d_values = np.linspace(defocus_range[0], defocus_range[1], 100)
        coc_values = [self.defocus_to_coc_px(d) for d in d_values]
        return (min(coc_values), max(coc_values))


def create_gaussian_kernel(sigma: float, radius_factor: float = 4.0) -> np.ndarray:
    """
    Create 2D Gaussian blur kernel.

    Args:
        sigma: Standard deviation of Gaussian
        radius_factor: Kernel radius as multiple of sigma

    Returns:
        Normalised 2D Gaussian kernel
    """
    if sigma <= 0:
        # Return identity kernel (no blur)
        return np.array([[1.0]])

    radius = int(np.ceil(radius_factor * sigma))
    size = 2 * radius + 1

    x = np.arange(size) - radius
    y = np.arange(size) - radius
    X, Y = np.meshgrid(x, y)

    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    return kernel.astype(np.float32)


def apply_gaussian_blur(
    image: np.ndarray,
    sigma: float,
    radius_factor: float = 4.0
) -> np.ndarray:
    """
    Apply Gaussian blur to image.

    Args:
        image: Input image (grayscale, float32, range [0, 1])
        sigma: Blur kernel standard deviation
        radius_factor: Kernel radius as multiple of sigma

    Returns:
        Blurred image
    """
    if sigma <= 0.05:
        return image.copy()

    kernel = create_gaussian_kernel(sigma, radius_factor)

    blurred = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    return blurred


def validate_sample_erf(
    blurred_image: np.ndarray,
    sigma_label: float,
    num_rays: int = 12,
) -> Dict[str, float]:
    """
    Fast ERF validation: measure actual Gaussian blur in a generated sample.

    Uses detect_sphere() for centre/radius, then a fast single-shot curve_fit
    per ray (seeded with sigma_label) instead of the full 10-start multi-start
    used in calibration. ~10-30× faster than measure_blur_erf().

    Args:
        blurred_image: Blurred sphere image (float32, [0,1], model scale)
        sigma_label: The training label sigma (model-scale pixels), used as
                     initial guess for curve_fit
        num_rays: Number of radial rays for ERF fitting (default 12)

    Returns:
        Dict with keys: sigma_measured_erf, erf_r_squared,
        erf_sigma_error_px, erf_sigma_error_pct.
        Values are float('nan') if detection or fitting fails.
    """
    from scipy.optimize import curve_fit
    from scipy.special import erf as sp_erf

    nan_result = {
        'sigma_measured_erf': float('nan'),
        'erf_r_squared': float('nan'),
        'erf_sigma_error_px': float('nan'),
        'erf_sigma_error_pct': float('nan'),
    }

    # Lazy import from calibration module
    try:
        from Calibration.blur_measurement import detect_sphere
    except ImportError:
        try:
            # Fallback for non-package usage
            import sys
            _calib_dir = str(Path(__file__).resolve().parent.parent / 'Calibration')
            if _calib_dir not in sys.path:
                sys.path.insert(0, _calib_dir)
            from blur_measurement import detect_sphere
        except ImportError:
            return nan_result

    # ERF edge model (inlined to avoid importing the full module each call)
    def _erf_edge(r, I_bg, I_sphere, r_edge, sigma):
        sigma = max(sigma, 0.001)
        return ((I_bg + I_sphere) / 2
                + (I_bg - I_sphere) / 2
                * sp_erf((r - r_edge) / (sigma * np.sqrt(2))))

    # Normalise to [0, 1]
    img = blurred_image
    if img.max() > 1:
        img = img.astype(np.float32) / 255.0

    # Detect sphere centre and radius
    centre, radius = detect_sphere(img)
    if centre is None or radius is None:
        return nan_result

    cx, cy = centre
    h, w = img.shape[:2]

    # Estimate contrast for initial guesses
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    inner = dist < radius * 0.7
    outer = (dist > radius * 1.3) & (dist < radius * 1.8)
    if np.any(inner) and np.any(outer):
        I_sphere_est = float(np.median(img[inner]))
        I_bg_est = float(np.median(img[outer]))
        contrast = abs(I_bg_est - I_sphere_est)
        if contrast < 0.05:
            return nan_result
    else:
        return nan_result

    # Sampling window around edge
    edge_margin = max(80, int(radius * 0.3))

    # Single-shot ERF fitting per ray, seeded with sigma_label
    sigmas = []
    r_squareds = []

    for i in range(num_rays):
        angle = 2 * np.pi * i / num_rays
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Per-ray max radius from image bounds
        if cos_a > 0.01:
            max_r_x = (w - 2 - cx) / cos_a
        elif cos_a < -0.01:
            max_r_x = (1 - cx) / cos_a
        else:
            max_r_x = float('inf')
        if sin_a > 0.01:
            max_r_y = (h - 2 - cy) / sin_a
        elif sin_a < -0.01:
            max_r_y = (1 - cy) / sin_a
        else:
            max_r_y = float('inf')
        max_r = min(max_r_x, max_r_y)

        start_r = max(0, radius - edge_margin)
        end_r = min(radius + edge_margin, max_r)
        if end_r <= start_r:
            continue

        r_values = np.arange(start_r, end_r, 0.5)
        if len(r_values) < 20:
            continue

        # Sub-pixel bilinear sampling
        x_coords = cx + r_values * cos_a
        y_coords = cy + r_values * sin_a
        x0 = np.floor(x_coords).astype(int)
        y0 = np.floor(y_coords).astype(int)
        x1, y1 = x0 + 1, y0 + 1
        valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
        if np.sum(valid) < 20:
            continue

        r_v = r_values[valid]
        xf = x_coords[valid] - x0[valid]
        yf = y_coords[valid] - y0[valid]
        intensities = (
            img[y0[valid], x0[valid]] * (1 - xf) * (1 - yf) +
            img[y0[valid], x1[valid]] * xf * (1 - yf) +
            img[y1[valid], x0[valid]] * (1 - xf) * yf +
            img[y1[valid], x1[valid]] * xf * yf
        )

        # 3-seed curve_fit: label value + bracket to catch large errors
        sigma_seeds = [
            max(sigma_label, 0.1),
            max(sigma_label * 0.3, 0.1),
            sigma_label * 3.0 + 1.0,
        ]
        best_popt, best_r_sq, best_res = None, -np.inf, np.inf
        bounds = ([0, 0, r_v.min(), 0.01], [1, 1, r_v.max(), 500])
        for s_init in sigma_seeds:
            try:
                popt, _ = curve_fit(
                    _erf_edge, r_v, intensities,
                    p0=[I_bg_est, I_sphere_est, radius, s_init],
                    bounds=bounds,
                    maxfev=500,
                )
                fitted = _erf_edge(r_v, *popt)
                res = float(np.sum((intensities - fitted) ** 2))
                if res < best_res:
                    best_popt, best_res = popt, res
                    ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
                    best_r_sq = 1 - (res / ss_tot) if ss_tot > 0 else 0
            except (RuntimeError, ValueError):
                continue

        if best_popt is not None and best_r_sq > 0.5 and best_popt[3] > 0.01:
            sigmas.append(best_popt[3])
            r_squareds.append(best_r_sq)

    if not sigmas:
        return nan_result

    sigma_measured = float(np.median(sigmas))
    confidence = float(np.mean(r_squareds))
    error_px = sigma_measured - sigma_label
    error_pct = (100.0 * error_px / sigma_label) if sigma_label != 0 else float('nan')

    return {
        'sigma_measured_erf': round(sigma_measured, 6),
        'erf_r_squared': round(confidence, 6),
        'erf_sigma_error_px': round(error_px, 6),
        'erf_sigma_error_pct': round(error_pct, 4),
    }


@dataclass
class SphereAppearanceStats:
    """Appearance statistics sampled from real sphere crops."""
    bg_mean: float          # Background intensity mean [0,1]
    bg_std: float           # Background intensity std across crops
    interior_mean: float    # Sphere interior intensity mean [0,1]
    interior_std: float     # Sphere interior intensity std across crops
    noise_std: float        # Sensor noise std [0,1] (from flat bg patch only)
    diameter_min: float     # Minimum detected diameter in pixels
    diameter_max: float     # Maximum detected diameter in pixels
    diameter_mean: float    # Mean detected diameter
    image_size: int         # Size of real crops (assumed square)
    has_highlight: bool     # Whether specular highlights were detected
    highlight_intensity: float  # Typical highlight brightness above interior
    vignette_strength: float    # Radial falloff: (centre - corner) intensity difference
    rim_light_strength: float   # Interior radial gradient: (edge - centre) difference

    @staticmethod
    def from_real_crops(
        image_paths: list,
        diameter_map: Optional[dict] = None,
    ) -> 'SphereAppearanceStats':
        """Scan real crops and extract appearance statistics."""
        def log(msg):
            print(msg)

        bg_intensities = []
        interior_intensities = []
        noise_stds = []
        diameters = []
        image_size = 128
        highlight_intensities = []
        vignette_strengths = []
        rim_light_strengths = []

        for img_path in image_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape
            image_size = max(h, w)
            img_f = img.astype(np.float32) / 255.0

            # Detect sphere via Otsu threshold
            _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bg_mask = mask > 128    # bright = background
            fg_mask = mask <= 128   # dark = sphere interior

            if bg_mask.sum() < 100 or fg_mask.sum() < 100:
                continue

            bg_val = float(np.median(img_f[bg_mask]))
            interior_val = float(np.median(img_f[fg_mask]))
            bg_intensities.append(bg_val)
            interior_intensities.append(interior_val)

            # Noise: measure from a flat corner patch well away from edges.
            # Use the top-left corner region (10% of image size) if it's all background.
            patch_size = max(16, h // 10)
            corner_patch = img_f[:patch_size, :patch_size]
            corner_mask = bg_mask[:patch_size, :patch_size]
            bg_fraction = corner_mask.sum() / corner_mask.size if corner_mask.size > 0 else 0
            if bg_fraction > 0.95:
                # Subtract local mean to remove any gradient, keep only noise
                smoothed = cv2.GaussianBlur(corner_patch, (5, 5), 0)
                residual = corner_patch - smoothed
                noise_stds.append(float(np.std(residual)))
            else:
                # Fallback: try bottom-right corner
                corner_patch = img_f[-patch_size:, -patch_size:]
                corner_mask = bg_mask[-patch_size:, -patch_size:]
                bg_fraction = corner_mask.sum() / corner_mask.size if corner_mask.size > 0 else 0
                if bg_fraction > 0.95:
                    smoothed = cv2.GaussianBlur(corner_patch, (5, 5), 0)
                    residual = corner_patch - smoothed
                    noise_stds.append(float(np.std(residual)))

            # Vignette: compare background intensity at centre vs corners
            cy_img, cx_img = h // 2, w // 2
            r_img = min(h, w) // 2
            yy, xx = np.mgrid[0:h, 0:w]
            dist_from_centre = np.sqrt((xx - cx_img)**2 + (yy - cy_img)**2)

            # Centre ring: background pixels within 20% of image radius
            centre_ring = bg_mask & (dist_from_centre < r_img * 0.2)
            # Corner ring: background pixels beyond 80% of image radius
            corner_ring = bg_mask & (dist_from_centre > r_img * 0.8)

            if centre_ring.sum() > 20 and corner_ring.sum() > 20:
                centre_intensity = float(np.mean(img_f[centre_ring]))
                corner_intensity = float(np.mean(img_f[corner_ring]))
                vignette_strengths.append(centre_intensity - corner_intensity)

            # Diameter from diameter_map or from mask
            if diameter_map and img_path.name in diameter_map:
                diameters.append(float(diameter_map[img_path.name]))
            else:
                area = fg_mask.sum()
                equiv_diam = 2.0 * np.sqrt(area / np.pi)
                diameters.append(equiv_diam)

            # Rim lighting: compare interior intensity near edge vs at centre.
            # Foreground pixels: split into outer 20% ring and inner core.
            fg_coords = np.argwhere(fg_mask)
            if len(fg_coords) > 100:
                fg_cy = fg_coords[:, 0].mean()
                fg_cx = fg_coords[:, 1].mean()
                fg_dist = np.sqrt((fg_coords[:, 1] - fg_cx)**2 + (fg_coords[:, 0] - fg_cy)**2)
                fg_radius = np.percentile(fg_dist, 95)
                if fg_radius > 10:
                    inner_mask_idx = fg_dist < fg_radius * 0.5
                    outer_mask_idx = fg_dist > fg_radius * 0.8
                    inner_vals = img_f[fg_coords[inner_mask_idx, 0], fg_coords[inner_mask_idx, 1]]
                    outer_vals = img_f[fg_coords[outer_mask_idx, 0], fg_coords[outer_mask_idx, 1]]
                    if len(inner_vals) > 10 and len(outer_vals) > 10:
                        rim_light_strengths.append(float(np.mean(outer_vals) - np.mean(inner_vals)))

            # Specular highlight: look for bright spot inside sphere
            interior_pixels = img_f[fg_mask]
            p95 = np.percentile(interior_pixels, 95)
            if p95 > interior_val + 0.1:
                highlight_intensities.append(p95 - interior_val)

        if not bg_intensities:
            log("Warning: Could not extract appearance stats from any real crop, using defaults")
            return SphereAppearanceStats(
                bg_mean=0.78, bg_std=0.03,
                interior_mean=0.05, interior_std=0.02,
                noise_std=0.003,
                diameter_min=30, diameter_max=100, diameter_mean=65,
                image_size=image_size,
                has_highlight=False, highlight_intensity=0.1,
                vignette_strength=0.02,
                rim_light_strength=0.01,
            )

        stats = SphereAppearanceStats(
            bg_mean=float(np.mean(bg_intensities)),
            bg_std=float(np.std(bg_intensities)) if len(bg_intensities) > 1 else 0.02,
            interior_mean=float(np.mean(interior_intensities)),
            interior_std=float(np.std(interior_intensities))
            if len(interior_intensities) > 1 else 0.01, noise_std=float(np.mean(noise_stds))
            if noise_stds else 0.003, diameter_min=float(np.min(diameters)),
            diameter_max=float(np.max(diameters)),
            diameter_mean=float(np.mean(diameters)),
            image_size=image_size, has_highlight=len(highlight_intensities) > len(image_paths) * 0.2,
            highlight_intensity=float(np.mean(highlight_intensities))
            if highlight_intensities else 0.1, vignette_strength=float(
                np.mean(vignette_strengths)) if vignette_strengths else 0.02,
            rim_light_strength=float(np.mean(rim_light_strengths))
            if rim_light_strengths else 0.01,)

        log(f"Sphere appearance stats from {len(bg_intensities)} crops:")
        log(f"  Background: {stats.bg_mean:.3f} +/- {stats.bg_std:.3f}"
            f"  (8-bit: {stats.bg_mean*255:.0f} +/- {stats.bg_std*255:.0f})")
        log(f"  Interior:   {stats.interior_mean:.3f} +/- {stats.interior_std:.3f}"
            f"  (8-bit: {stats.interior_mean*255:.0f} +/- {stats.interior_std*255:.0f})")
        log(f"  Noise std:  {stats.noise_std:.5f}  (8-bit: {stats.noise_std*255:.1f})")
        log(f"  Diameters:  {stats.diameter_min:.0f} - {stats.diameter_max:.0f} px "
            f"(mean {stats.diameter_mean:.0f})")
        log(f"  Highlights: {'detected' if stats.has_highlight else 'none'} "
            f"(intensity {stats.highlight_intensity:.3f})")
        log(f"  Vignette:   {stats.vignette_strength:.4f}  "
            f"(centre-corner delta, 8-bit: {stats.vignette_strength*255:.1f})")
        log(f"  Rim light:  {stats.rim_light_strength:.4f}  "
            f"(edge-centre delta inside sphere)")

        return stats


def generate_realistic_sphere(
    diameter_px: int,
    image_size: int = 128,
    stats: Optional[SphereAppearanceStats] = None,
    centre: Optional[Tuple[int, int]] = None,
    add_highlight: bool = True,
    add_vignette: bool = True,
    add_rim_light: bool = True,
) -> np.ndarray:
    """
    Generate a realistic synthetic sphere silhouette matching real crop appearance.

    Args:
        diameter_px: Sphere diameter in pixels at output resolution.
        image_size: Output image size (square).
        stats: Appearance statistics from real crops. Falls back to defaults if None.
        centre: Optional (x, y) centre. Random jitter around image centre if None.
        add_highlight: Whether to probabilistically add a specular highlight.
        add_vignette: Whether to add radial background vignette.
        add_rim_light: Whether to add interior radial gradient (rim lighting).

    Returns:
        Synthetic sphere image, float32 [0,1], shape (image_size, image_size).
    """
    if stats is None:
        # Minimal fallback: plain circle on white background
        img = np.ones((image_size, image_size), dtype=np.float32)
        c = centre if centre is not None else (image_size // 2, image_size // 2)
        cv2.circle(img, c, diameter_px // 2, 0.0, -1)
        return img

    rng = np.random.default_rng()

    # Sample appearance for this image
    bg_val = np.clip(rng.normal(stats.bg_mean, stats.bg_std), 0.3, 1.0)
    interior_val = np.clip(rng.normal(stats.interior_mean, stats.interior_std), 0.0, bg_val - 0.1)

    # Centre with small random jitter
    if centre is None:
        jitter = int(image_size * 0.02)
        cx = image_size // 2 + rng.integers(-jitter, jitter + 1)
        cy = image_size // 2 + rng.integers(-jitter, jitter + 1)
    else:
        cx, cy = centre

    radius = diameter_px / 2.0

    # Build anti-aliased sphere mask using distance field
    y, x = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    dist_from_centre = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Hard edge — native blur is applied separately after rendering so that
    # the exact sigma is known for quadrature subtraction.
    alpha = (dist_from_centre > radius).astype(np.float32)
    # alpha=1 is background, alpha=0 is interior

    # Base image: flat background + flat interior
    image = alpha * bg_val + (1.0 - alpha) * interior_val

    # Radial vignette on background: brighter at centre, darker at corners
    if add_vignette and stats.vignette_strength > 0.002:
        max_dist = np.sqrt(2.0) * (image_size / 2.0)  # distance to corner
        vignette_magnitude = rng.uniform(
            stats.vignette_strength * 0.5, stats.vignette_strength * 1.5,
        )
        # Normalised radial distance [0 at centre, 1 at corner]
        r_norm = dist_from_centre / max_dist
        # Quadratic falloff: 0 at centre, -vignette_magnitude at corners
        vignette = -vignette_magnitude * r_norm ** 2
        # Apply only to background
        image += vignette * alpha

    # Interior rim lighting: slightly lighter near edge, darker at centre
    if add_rim_light and stats.rim_light_strength > 0.002:
        rim_magnitude = rng.uniform(
            stats.rim_light_strength * 0.5, stats.rim_light_strength * 1.5,
        )
        # Normalised distance from sphere centre [0 at centre, 1 at edge]
        r_sphere = np.clip(dist_from_centre / max(radius, 1.0), 0.0, 1.0)
        # Quadratic ramp: 0 at centre, rim_magnitude at edge
        rim_gradient = rim_magnitude * r_sphere ** 2
        # Apply only inside sphere
        image += rim_gradient * (1.0 - alpha)

    # Specular highlight (probabilistic)
    if add_highlight and stats.has_highlight and rng.random() < 0.3:
        hl_offset_x = rng.normal(0, radius * 0.15)
        hl_offset_y = rng.normal(0, radius * 0.15)
        hl_cx = cx + hl_offset_x
        hl_cy = cy + hl_offset_y
        hl_sigma = rng.uniform(radius * 0.05, radius * 0.15)
        hl_intensity = rng.uniform(0.5, 1.0) * stats.highlight_intensity

        hl_dist2 = (x - hl_cx) ** 2 + (y - hl_cy) ** 2
        highlight = hl_intensity * np.exp(-hl_dist2 / (2 * hl_sigma ** 2))
        image += highlight * (1.0 - alpha)

    # Sensor noise (measured from flat background patch, low magnitude)
    if stats.noise_std > 0:
        noise = rng.normal(0, stats.noise_std, (image_size, image_size)).astype(np.float32)
        image += noise

    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    return image


class SyntheticBlurGenerator:
    """
    Generate synthetic blurred training data from sharp droplet images.
    """

    def __init__(
        self,
        optical_params: BlurParams,
        defocus_range_mm: Tuple[float, float] = (-12.0, 12.0),
        coc_range_px: Optional[Tuple[float, float]] = None,
        image_size: int = 128,
        crop_size: Optional[int] = None,
        calibration_reference_resolution: Optional[int] = None,
        radius_factor: float = 4.0,
        blur_distribution: str = "uniform",
        beta_alpha: float = 2.0,
        beta_beta: float = 5.0,
        min_blur_px: Optional[float] = None,
        calibration_model=None,  # type: Optional[Any]
    ):
        """
        Args:
            optical_params: Optical system parameters
            defocus_range_mm: Physical defocus range to sample from
            coc_range_px: Optional direct CoC range at NATIVE scale (overrides defocus_range)
            image_size: Output image size (model_size, e.g., 128 or 256)
            crop_size: Actual crop size on disk (e.g., 299). Used for native blur scaling.
                       If None, defaults to image_size.
            calibration_reference_resolution: Resolution at which rho_direct was measured
                       (e.g., 860). Used for sigma label/normalization scaling in direct mode.
                       If None, falls back to crop_size.
            radius_factor: Gaussian kernel radius factor
            blur_distribution: "uniform" or "weighted" - how to sample blur values
            beta_alpha: Beta distribution alpha parameter (for weighted sampling)
            beta_beta: Beta distribution beta parameter (for weighted sampling)
            min_blur_px: Optional minimum blur threshold at calibration native scale
            calibration_model: Optional ``physics.CalibrationModel`` instance.
                When provided, ``defocus_to_sigma`` and ``_save_tertiary_sample``
                delegate to this model for σ↔z math (supports linear /
                quadrature / hybrid methods). When None, the legacy
                linear formula via ``optical_params.rho_direct`` /
                ``optical_params.sigma_0`` is used (bit-identical to
                pre-Phase-5 behaviour). New datasets should pass this in
                so the chosen calibration method propagates downstream.
        """
        self.params = optical_params
        self.blur_calc = BlurCalculator(optical_params,
                                          calibration_model=calibration_model)
        self.image_size = image_size
        self.crop_size = crop_size if crop_size is not None else image_size
        self.radius_factor = radius_factor
        self.blur_distribution = blur_distribution
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.min_blur_px = min_blur_px
        # Phase 5: optional CalibrationModel for unified σ↔z math.
        # When None, legacy linear formula is used (back-compat).
        self.calibration_model = calibration_model
        self.sphere_stats: Optional[SphereAppearanceStats] = None  # Set by generate_dataset
        # When set (by generate_dataset), synthesis (sphere rendering + blur)
        # happens at this resolution before a final resize to image_size.
        # If None, synthesis happens directly at image_size (existing behaviour).
        self.synthesis_size: Optional[int] = None

        # resolution_scale: for optical mode only — scales CoC range from calib-native to model-scale.
        # Direct mode uses native_to_model_scale (crop→model); see generate_sample().
        calib_res = calibration_reference_resolution if calibration_reference_resolution is not None else self.crop_size
        self.resolution_scale = self.image_size / calib_res
        self.calibration_reference_resolution = calib_res
        # native_to_model_scale: converts crop-space blur (native_blur_sigma from CSV) to model-scale
        self.native_to_model_scale = self.image_size / self.crop_size

        # Determine blur range at NATIVE scale first (optical mode only)
        self.defocus_range = defocus_range_mm
        if self.params.training_mode == "direct":
            # Direct mode doesn't use CoC range — set placeholder
            coc_range_native = (0.0, 0.0)
        elif coc_range_px is not None:
            coc_range_native = coc_range_px
        else:
            coc_range_native = self.blur_calc.get_coc_range(defocus_range_mm)

        # Apply minimum blur if specified (at native scale, optical mode)
        if min_blur_px is not None and min_blur_px > 0:
            coc_min_filtered = max(coc_range_native[0], min_blur_px)
            coc_range_native = (coc_min_filtered, coc_range_native[1])

        # Store native range for reference
        self.coc_range_native = coc_range_native

        # Scale to model size - THIS IS THE KEY CHANGE
        # Labels and blur are now at model scale, not native scale
        self.coc_range = (
            coc_range_native[0] * self.resolution_scale,
            coc_range_native[1] * self.resolution_scale
        )

        # For direct mode: pre-compute max sigma at calibration scale.
        # The final max_sigma (at model scale) is computed in generate_dataset()
        # once we know the actual training crops and their per-camera scales.
        if self.params.training_mode == "direct":
            max_defocus = max(abs(self.defocus_range[0]), abs(self.defocus_range[1]))
            self.max_sigma_calib = self.blur_calc.defocus_to_sigma(max_defocus)
            # Fallback until generate_dataset() recomputes with cross-camera correction
            self.max_sigma = self.max_sigma_calib * self.native_to_model_scale
            # Will be set in generate_dataset() once we know the actual training crops
            self.min_sigma_model = None

        # Store for later logging (direct mode info is logged in generate_dataset instead)
        if self.params.training_mode == "direct":
            self._init_info = None
        else:
            self._init_info = (
                f"CoC range (model scale): {self.coc_range[0]:.2f} - {self.coc_range[1]:.2f} px, "
                f"CoC range (native): {self.coc_range_native[0]:.1f} - {self.coc_range_native[1]:.1f} px, "
                f"Resolution scale: {self.resolution_scale:.3f} ({self.crop_size}→{self.image_size})")

    def blur_image(
        self,
        sharp_image: np.ndarray,
        coc_px: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply synthetic defocus blur to sharp image.

        Args:
            sharp_image: Sharp input image (0 = droplet, 1 = background)
            coc_px: Circle of confusion in pixels

        Returns:
            Tuple of (blurred_image, coc_map)
        """
        sigma = self.blur_calc.coc_to_sigma(coc_px)
        blurred = apply_gaussian_blur(sharp_image, sigma, self.radius_factor)

        # Create CoC map (uniform value across image)
        coc_map = np.full_like(sharp_image, coc_px / self.coc_range[1])  # Normalise to [0, 1]

        return blurred, coc_map

    def _sample_in_range(self, lo: float, hi: float) -> float:
        """Sample a value in [lo, hi] using weighted (beta) or uniform distribution."""
        if hi <= lo:
            return 0.0
        if self.blur_distribution == "weighted":
            beta_sample = np.random.beta(a=self.beta_alpha, b=self.beta_beta)
            return lo + beta_sample * (hi - lo)
        return random.uniform(lo, hi)

    def generate_sample(
        self,
        sharp_image: Optional[np.ndarray] = None,
        diameter_px: Optional[int] = None,
        coc_px: Optional[float] = None,
        scale_px_per_mm: Optional[float] = None,
        native_blur_sigma: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a single training sample.

        In optical mode: samples CoC, applies blur, normalizes by max CoC.
        In direct mode: samples defocus, computes sigma via defocus_to_sigma(),
        applies blur, normalizes by max sigma.

        Args:
            sharp_image: Optional pre-existing sharp image
            diameter_px: Droplet diameter (if generating synthetic)
            coc_px: Optional specific CoC value, otherwise random (optical mode only)
            scale_px_per_mm: Camera scale for this image (px/mm). If provided and
                scale_calib_px_per_mm is set in params, applies cross-camera correction
                to sigma so blur matches the inference camera scale.
            native_blur_sigma: Measured Gaussian sigma already present in the sharp
                crop (px, at crop resolution). Used in direct mode to compute the
                quadrature kernel so the output has exactly sigma_defocus total blur.

        Returns:
            Dict with 'sharp', 'blur', 'defocus_mm', and mode-specific keys:
              - Direct mode: 'sigma_map', 'sigma_value', 'sigma_kernel'
              - Optical mode: 'coc_map', 'coc_value'
        """
        # Generate or use provided sharp image
        # Render-target resolution: use synthesis_size if set, else image_size.
        render_size = self.synthesis_size if self.synthesis_size else self.image_size
        if sharp_image is None:
            if diameter_px is None:
                if self.sphere_stats is not None:
                    # Scale diameters from native crop resolution to render resolution
                    scale = render_size / self.sphere_stats.image_size
                    d_min = max(int(self.sphere_stats.diameter_min * scale), render_size // 5)
                    d_max = min(
                        int(self.sphere_stats.diameter_max * scale),
                        render_size * 4 // 5)
                    if d_min >= d_max:
                        d_min = d_max - 1
                    diameter_px = random.randint(d_min, d_max)
                else:
                    diameter_px = random.randint(10, 50)
            sharp_image = generate_realistic_sphere(
                diameter_px, render_size, stats=self.sphere_stats,
            )
            # Apply native blur to synthetic sphere so it matches real crop
            # edge character.  The caller passes the sampled native_blur_sigma
            # and quadrature subtraction accounts for it exactly.
            if native_blur_sigma > 0:
                from scipy.ndimage import gaussian_filter
                # native_blur_sigma is in crop-pixel units; convert to model px,
                # then up-scale to render_size if synthesis happens there.
                sigma_at_model = native_blur_sigma * self.native_to_model_scale
                if render_size != self.image_size:
                    sigma_at_render = sigma_at_model * (render_size / self.image_size)
                else:
                    sigma_at_render = sigma_at_model
                sharp_image = gaussian_filter(sharp_image, sigma=sigma_at_render)

        mode = getattr(self.params, 'training_mode', 'optical')

        if mode == "direct":
            # Direct mode: sample sigma then back-compute defocus.
            # Uniform defocus sampling would give 2× density near zero (both ± sides contribute),
            # so we sample sigma directly and assign sign proportionally to the range extents.
            rho = self.params.rho_direct
            sigma_0 = self.params.sigma_0 if self.params.sigma_0 is not None else 0.0
            scale_calib = self.params.scale_calib_px_per_mm

            # Phase 5: prefer the CalibrationModel for σ↔z math when available.
            # Falls back to legacy linear formula if model not set (back-compat).
            cm = getattr(self, 'calibration_model', None)

            def _sigma_calib_to_abs_z(sigma_calib_local: float) -> float:
                """σ_calib → |z|_mm.  Method-aware when CalibrationModel set."""
                if cm is not None:
                    z, _flag = cm.inverse(sigma_calib_local)
                    return float(z) if z == z else 0.0  # nan-safe
                if rho and rho > 0:
                    return max((sigma_calib_local - sigma_0) / rho, 0.0)
                return 0.0

            # Determine cross-camera correction factor for this crop
            cc_factor = 1.0
            if scale_px_per_mm is not None and scale_calib is not None and scale_calib > 0:
                cc_factor = scale_px_per_mm / scale_calib

            # Sample in model space if min_sigma_model was computed in generate_dataset().
            # This ensures uniform distribution at model scale regardless of per-crop
            # cross-camera correction, fixing last-bin underpopulation.
            if hasattr(self, 'min_sigma_model') and self.min_sigma_model is not None:
                min_model = self.min_sigma_model
                max_model = self.max_sigma

                sigma_model = self._sample_in_range(min_model, max_model)

                # Back-compute calibration-space sigma and physical defocus
                sigma_calib = sigma_model / (cc_factor * self.native_to_model_scale)
                abs_defocus = _sigma_calib_to_abs_z(sigma_calib)
            else:
                # Fallback: sample in calibration space (no scale_map available)
                max_defocus_mag = max(abs(self.defocus_range[0]), abs(self.defocus_range[1]))
                # Forward bound: use calibration model when present (correct
                # for quadrature/hybrid where σ = √((ρz)² + floor²) ≠ ρz + σ_0).
                if cm is not None:
                    max_sigma_calib = float(cm.forward(max_defocus_mag))
                    min_sigma_calib = float(cm.forward(0.0))
                else:
                    max_sigma_calib = (rho * max_defocus_mag + sigma_0) if rho else 0.0
                    min_sigma_calib = sigma_0
                if self.min_blur_px is not None and self.min_blur_px > min_sigma_calib:
                    min_sigma_calib = self.min_blur_px

                sigma_defocus = self._sample_in_range(min_sigma_calib, max_sigma_calib)
                abs_defocus = _sigma_calib_to_abs_z(sigma_defocus)
                sigma_defocus *= cc_factor
                sigma_model = sigma_defocus * self.native_to_model_scale

            neg_span = abs(min(self.defocus_range[0], 0.0))
            pos_span = max(self.defocus_range[1], 0.0)
            total_span = neg_span + pos_span
            p_neg = neg_span / total_span if total_span > 0 else 0.5
            defocus_mm = -abs_defocus if random.random() < p_neg else abs_defocus

            # Quadrature kernel: the crop already has native_blur_sigma baseline blur,
            # so we apply only the remaining blur needed to reach sigma_model.
            # native_blur_sigma is in crop-pixel units → use native_to_model_scale (128/299).
            # sigma_model above also uses native_to_model_scale — both terms are in model-px.
            sigma_native_model = native_blur_sigma * self.native_to_model_scale
            kernel_sq = sigma_model ** 2 - sigma_native_model ** 2
            if kernel_sq > 0:
                sigma_kernel = float(np.sqrt(kernel_sq))
            else:
                # Native blur already exceeds target — no kernel applied.
                # Clamp label to the actual blur in the image so label matches content.
                sigma_kernel = 0.0
                sigma_model = sigma_native_model

            # If synthesis happens at higher resolution than the model's
            # final input, scale the blur kernel up so that after the
            # downsample-to-model-size step at the end, the effective
            # blur in the saved image is sigma_kernel as labelled.
            if self.synthesis_size and self.synthesis_size != self.image_size:
                synth_scale = self.synthesis_size / self.image_size
                sigma_for_blur = sigma_kernel * synth_scale
            else:
                sigma_for_blur = sigma_kernel

            blurred = apply_gaussian_blur(sharp_image, sigma_for_blur, self.radius_factor)

            # Normalized blur map (clamp to [0, 1] in case native blur exceeds max_sigma)
            normalized_blur = min(sigma_model / self.max_sigma, 1.0) if self.max_sigma > 0 else 0.0
            sigma_map = np.full_like(sharp_image, normalized_blur)

            # Downsample to model size if synthesis was at higher resolution
            if self.synthesis_size and self.synthesis_size != self.image_size:
                ms = self.image_size
                sharp_image = cv2.resize(sharp_image, (ms, ms),
                                          interpolation=cv2.INTER_AREA)
                blurred = cv2.resize(blurred, (ms, ms),
                                      interpolation=cv2.INTER_AREA)
                sigma_map = cv2.resize(sigma_map, (ms, ms),
                                        interpolation=cv2.INTER_AREA)

            return {
                'sharp': sharp_image,
                'blur': blurred,
                'sigma_map': sigma_map,
                'sigma_value': sigma_model,  # sigma at model scale (px)
                'defocus_mm': defocus_mm,
                'sigma_kernel': sigma_kernel,   # actual Gaussian applied — for blur trace audit
            }
        else:
            # Optical mode: existing CoC-based workflow
            if coc_px is None:
                coc_px = self._sample_in_range(self.coc_range[0], self.coc_range[1])

            # Apply blur
            blurred, coc_map = self.blur_image(sharp_image, coc_px)

            # Calculate corresponding defocus distance
            defocus_mm = self.blur_calc.blur_to_defocus(coc_px)

            return {
                'sharp': sharp_image,
                'blur': blurred,
                'coc_map': coc_map,
                'coc_value': coc_px,
                'defocus_mm': defocus_mm
            }

    def generate_dataset(
        self,
        output_dir: Union[str, Path],
        num_samples: int = 50000,
        sharp_images_dir: Optional[Union[str, Path]] = None,
        diameter_range_px: Tuple[int, int] = (10, 50),
        add_noise: bool = False,
        noise_level: float = 0.01,
        camera_filter: str = "all",
        save_blur_trace: bool = False,
        erf_validation: bool = False,
        erf_validation_count: Optional[int] = None,
        synthetic_fraction: Optional[float] = None,
        synthetic_source_size: Optional[int] = None,
        secondary_source_dir: Optional[Union[str, Path]] = None,
        secondary_source_size: Optional[int] = None,
        secondary_fraction: float = 0.0,
        tertiary_source_dir: Optional[Union[str, Path]] = None,
        tertiary_fraction: float = 0.0,
    ) -> dict:
        """
        Generate full training dataset.

        Args:
            output_dir: Directory to save generated data
            num_samples: Number of samples to generate
            sharp_images_dir: Optional directory with real sharp crops
            diameter_range_px: Range of synthetic droplet diameters
            add_noise: Whether to add Gaussian noise
            noise_level: Noise standard deviation (if add_noise)
            camera_filter: Which camera subfolders to use ("all", "g", "m", "v")
            save_blur_trace: If True and training_mode == 'direct', appends blur
                trace columns to metadata.csv. Default False = identical to existing
                behaviour.
            erf_validation: If True, run ERF edge fitting on a subset of samples
                to measure actual blur and compare against labels.
            erf_validation_count: Number of samples to validate with ERF. If None
                or >= num_samples, validates all. Samples are evenly spaced.

        Returns:
            dict with generation metadata:
                - 'diameter_bins_used': bool (whether stratified sampling was used)
                - 'diameter_bin_boundaries': tuple (p33, p67) if stratified, else None
        """
        def log(msg):
            print(msg)

        # Log generator settings
        if hasattr(self, '_init_info') and self._init_info is not None:
            log(self._init_info)

        # Log minimum blur if applied
        if self.min_blur_px is not None and self.min_blur_px > 0:
            blur_label = "blur" if self.params.training_mode == "direct" else "CoC"
            log(f"Minimum {blur_label} filter applied: {self.min_blur_px:.2f} px (at calibration scale)")
            if self.params.training_mode != "direct":
                log(
                    f"   {blur_label} range adjusted to [{self.coc_range[0]:.2f}, {self.coc_range[1]:.2f}] px")

        # Configure synthesis_size for the PRIMARY source.
        # Synthesis_size is set per-sample below so secondary-source
        # samples can use a different resolution. The starting value here
        # is the primary one; the loop swaps it per sample as needed.
        primary_synthesis_size = (int(synthetic_source_size)
                                   if synthetic_source_size and synthetic_source_size > 0
                                   else None)
        secondary_synthesis_size = (int(secondary_source_size)
                                     if secondary_source_size and secondary_source_size > 0
                                     else None)
        self.synthesis_size = primary_synthesis_size
        if primary_synthesis_size:
            log(f"Primary synthesis size = {primary_synthesis_size} px "
                f"(blur at this resolution, resized to {self.image_size} for save)")

        # Load SECONDARY pool of sharp templates (for two-source mixing).
        # Used when secondary_fraction > 0 AND secondary_source_dir set.
        secondary_sharps: list[Path] = []
        secondary_stats: Optional[SphereAppearanceStats] = None
        # Calibration camera scale for secondary samples — looked up from
        # the bundle's calibration_results.yaml (or the parent's). Falls
        # back to self.params.scale_calib_px_per_mm if unavailable.
        secondary_scale_px_per_mm: Optional[float] = None
        sec_frac = max(0.0, min(1.0, float(secondary_fraction or 0.0)))
        if sec_frac > 0 and secondary_source_dir is not None:
            sec_dir = Path(secondary_source_dir)
            if sec_dir.is_dir():
                secondary_sharps = sorted(sec_dir.glob("*.png"))
                if secondary_sharps:
                    log(f"Secondary source: {len(secondary_sharps)} PNGs from {sec_dir}")
                    if secondary_synthesis_size:
                        log(f"Secondary synthesis size = {secondary_synthesis_size} px")
                    # Cache appearance stats so rendered spheres for
                    # secondary samples reflect the secondary source.
                    try:
                        secondary_stats = SphereAppearanceStats.from_real_crops(
                            secondary_sharps[:50])
                    except Exception as e:
                        log(f"WARNING: failed to extract secondary stats: {e}")
                    # Look up calibration scale from the bundle. The
                    # convention (calibration GUI bundle layout) is:
                    #   <bundle>/processed_images/*.png
                    #   <bundle>/calibration_results.yaml
                    # If user pointed at processed_images/ directly, the
                    # yaml is at the parent.
                    yaml_candidates = [
                        sec_dir / "calibration_results.yaml",
                        sec_dir.parent / "calibration_results.yaml",
                    ]
                    for yp in yaml_candidates:
                        if yp.is_file():
                            try:
                                import yaml as _yaml
                                with open(yp, "r") as _f:
                                    _cfg = _yaml.safe_load(_f) or {}
                                # Direct mode: scale_calib_px_per_mm at top
                                # of `direct` block; some setups put it at
                                # root. Try both.
                                _scale = (
                                    _cfg.get("direct", {}).get(
                                        "scale_calib_px_per_mm")
                                    or _cfg.get("scale_calib_px_per_mm")
                                )
                                if _scale:
                                    secondary_scale_px_per_mm = float(_scale)
                                    log(
                                        f"Secondary calibration scale: "
                                        f"{secondary_scale_px_per_mm:.4f} px/mm "
                                        f"(from {yp.name})")
                                break
                            except Exception as e:
                                log(f"WARNING: could not read {yp}: {e}")
                    if secondary_scale_px_per_mm is None:
                        # Fall back to the training config's calibration scale
                        secondary_scale_px_per_mm = (
                            self.params.scale_calib_px_per_mm)
                        if secondary_scale_px_per_mm:
                            log(
                                f"Secondary calibration scale (fallback "
                                f"to training config): "
                                f"{secondary_scale_px_per_mm:.4f} px/mm")
                else:
                    log(f"WARNING: secondary source {sec_dir} has no PNGs — "
                        f"secondary mixing disabled")
                    sec_frac = 0.0
            else:
                log(f"WARNING: secondary_source_dir {sec_dir} not found — "
                    f"secondary mixing disabled")
                sec_frac = 0.0

        # Load TERTIARY pool — REAL calibration PNGs with ERF-measured sigma
        # labels (no Gaussian blur applied; the image IS the blur, the
        # label measures it). Used for direct supervised learning on real
        # calibration data.
        # tertiary_pool: list of (PIL-readable Path, source_size, ERF_sigma_at_source_px)
        tertiary_pool: list[tuple[Path, int, float]] = []
        ter_frac = max(0.0, min(1.0, float(tertiary_fraction or 0.0)))
        if ter_frac > 0 and tertiary_source_dir is not None:
            import pandas as pd  # local import — module-level may not be loaded
            ter_dir = Path(tertiary_source_dir)
            # Find measurements.csv (in dir or its parent — bundle convention)
            csv_candidates = [
                ter_dir / "measurements.csv",
                ter_dir.parent / "measurements.csv",
            ]
            csv_path = next((c for c in csv_candidates if c.is_file()), None)
            if not ter_dir.is_dir():
                log(f"WARNING: tertiary_source_dir {ter_dir} not found — "
                    f"tertiary mixing disabled")
                ter_frac = 0.0
            elif csv_path is None:
                log(f"WARNING: no measurements.csv near {ter_dir} — "
                    f"tertiary mixing disabled (need ERF labels)")
                ter_frac = 0.0
            else:
                try:
                    csv_df = pd.read_csv(csv_path)
                    if "filename" not in csv_df.columns or "sigma_px" not in csv_df.columns:
                        raise ValueError("measurements.csv missing filename/sigma_px columns")
                    # Determine source size by reading one frame
                    first_png = next(ter_dir.glob("*.png"), None)
                    if first_png is None:
                        raise FileNotFoundError(f"No PNGs in {ter_dir}")
                    _probe = cv2.imread(str(first_png), cv2.IMREAD_GRAYSCALE)
                    ter_source_size = max(_probe.shape[:2])
                    # Build pool
                    for _, r in csv_df.iterrows():
                        fn = str(r["filename"])
                        sigma = r["sigma_px"]
                        if pd.isna(sigma):
                            continue
                        png_path = ter_dir / fn
                        if not png_path.is_file():
                            continue
                        tertiary_pool.append((png_path, ter_source_size, float(sigma)))
                    if not tertiary_pool:
                        log(f"WARNING: tertiary pool is empty (no matching PNG/CSV "
                            f"rows in {ter_dir}) — tertiary mixing disabled")
                        ter_frac = 0.0
                    else:
                        log(f"Tertiary source: {len(tertiary_pool)} real calibration "
                            f"PNGs from {ter_dir} (source_size={ter_source_size}px, "
                            f"sigma range {min(s for _,_,s in tertiary_pool):.2f}-"
                            f"{max(s for _,_,s in tertiary_pool):.2f} source-px)")
                except Exception as e:
                    log(f"WARNING: could not load tertiary pool from {ter_dir}: {e}")
                    ter_frac = 0.0
        # Cache the primary stats reference for swap-back after secondary samples
        primary_sphere_stats = None  # set after self.sphere_stats is populated

        # ERF validation setup
        validate_indices = set()
        if erf_validation:
            n_validate = num_samples
            if erf_validation_count is not None and erf_validation_count < num_samples:
                n_validate = max(1, erf_validation_count)
            step = max(1, num_samples // n_validate)
            validate_indices = set(range(0, num_samples, step))
            log(f"ERF validation enabled: {len(validate_indices)} samples "
                f"(every {step}{' sample' if step == 1 else ' samples'})")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / 'sharp').mkdir(exist_ok=True)
        (output_dir / 'blur').mkdir(exist_ok=True)
        (output_dir / 'blur_map').mkdir(exist_ok=True)

        # Scale diameter range based on image size (fallback for old behavior)
        min_diam = max(diameter_range_px[0], self.image_size // 5)   # 20% of image
        max_diam = min(diameter_range_px[1], self.image_size * 3 // 5)  # 60% of image
        diameter_range_px = (min_diam, max_diam)

        # Load real sharp images and organize by diameter bins
        diameter_bins = None  # Will store {bin_idx: [(path, diameter), ...]}
        bin_medians = None  # Will store [median_diam_bin0, median_diam_bin1, median_diam_bin2]
        bin_boundaries = None  # Will store (p33, p67) tertile boundaries
        use_binning = False
        real_sharps = []  # Fallback list for old behavior
        scale_map = {}        # filename -> scale_px_per_mm (for cross-camera correction)
        native_blur_map = {}  # filename -> native_blur_sigma in crop pixels
        camera_map = {}       # filename -> camera label

        if sharp_images_dir is not None:
            sharp_dir = Path(sharp_images_dir)

            # Try to load sharp_crops.csv for diameter information
            csv_path = sharp_dir / 'sharp_crops.csv'
            diameter_map = {}  # filename -> diameter_px

            if csv_path.exists():
                import pandas as pd
                try:
                    df = pd.read_csv(csv_path)
                    if 'filename' not in df.columns:
                        log("Warning: 'filename' column missing from sharp_crops.csv — "
                            "scale/native_blur lookups will be empty (cross-camera correction disabled)")
                    if 'filename' in df.columns and 'diameter_px' in df.columns:
                        diameter_map = dict(zip(df['filename'], df['diameter_px']))
                        log(f"Loaded diameter info for {len(diameter_map)} crops from CSV")
                    if 'filename' in df.columns and 'scale_px_per_mm' in df.columns:
                        scale_map = dict(zip(df['filename'], df['scale_px_per_mm']))
                        log(f"Loaded scale info for {len(scale_map)} crops from CSV")
                    if 'filename' in df.columns and 'native_blur_sigma' in df.columns:
                        native_blur_map = dict(
                            zip(df['filename'], df['native_blur_sigma'].fillna(0.0)))
                        log(f"Loaded native blur info for {len(native_blur_map)} crops from CSV")
                    if 'filename' in df.columns and 'camera' in df.columns:
                        camera_map = dict(zip(df['filename'], df['camera'].astype(str)))
                except Exception as e:
                    log(f"Warning: Could not read sharp_crops.csv: {e}")

            # NOTE: max_sigma update moved AFTER image scanning (see below)
            # so it uses only the scales of crops that will actually be trained on.

            # Scan for image files - supports nested structure: root/material/camera/images
            material_folders = [d for d in sharp_dir.iterdir() if d.is_dir()]

            if material_folders:
                for material_folder in material_folders:
                    # Check if material folder has camera subfolders (g, m, v)
                    camera_folders = [d for d in material_folder.iterdir() if d.is_dir()]

                    if camera_folders:
                        # Nested structure: material/camera/images
                        for camera_folder in camera_folders:
                            # Filter by camera type if specified
                            if camera_filter and camera_filter != "all":
                                if camera_folder.name.lower() != camera_filter.lower():
                                    continue

                            for ext in ['*.png', '*.jpg', '*.tif', '*.bmp']:
                                real_sharps.extend(list(camera_folder.glob(ext)))
                    else:
                        # Flat structure: material folder contains images directly
                        # Filter by folder name if it matches camera filter
                        if camera_filter and camera_filter != "all":
                            if material_folder.name.lower() != camera_filter.lower():
                                continue

                        for ext in ['*.png', '*.jpg', '*.tif', '*.bmp']:
                            real_sharps.extend(list(material_folder.glob(ext)))
            else:
                # No subfolders - scan root directly (single folder case)
                for ext in ['*.png', '*.jpg', '*.tif', '*.bmp']:
                    real_sharps.extend(list(sharp_dir.glob(ext)))

            if camera_filter and camera_filter != "all":
                log(f"Filtered to camera '{camera_filter}'")

            log(f"Found {len(real_sharps)} real sharp images")

            # --- Recompute max_sigma using only the ACTUAL training crops' scales ---
            # The provisional max_sigma from __init__ used calibration-scale sigma
            # without the cross-camera correction (s_c/s_calib). Now that we know
            # which images will be trained on, compute the correct value.
            if (self.params.training_mode == "direct"
                    and scale_map
                    and self.params.scale_calib_px_per_mm is not None
                    and self.params.scale_calib_px_per_mm > 0):
                # Filter scale_map to only the files that will actually be used
                training_filenames = {p.name for p in real_sharps}
                training_scales = [
                    float(scale_map[fn])
                    for fn in training_filenames
                    if fn in scale_map and np.isfinite(float(scale_map[fn]))
                ]

                if training_scales:
                    max_training_scale = max(training_scales)
                    scale_calib = self.params.scale_calib_px_per_mm
                    scale_ratio = max_training_scale / scale_calib
                    max_defocus = max(abs(self.defocus_range[0]), abs(self.defocus_range[1]))
                    sigma_0 = self.params.sigma_0 if self.params.sigma_0 is not None else 0.0
                    rho = self.params.rho_direct

                    self.max_sigma = (self.max_sigma_calib
                                      * scale_ratio
                                      * self.native_to_model_scale)

                    # Compute min_sigma_model for model-space sampling.
                    # Use the LOWEST cc_factor crop so the floor is the true minimum
                    # reachable in model space across all training crops.
                    min_training_scale = min(training_scales)
                    min_scale_ratio = min_training_scale / scale_calib
                    # Phase 11: use the calibration's actual minimum σ via the
                    # CalibrationModel (forward(0)). For linear this equals
                    # σ_0; for quadrature/hybrid this equals σ_floor — the
                    # diffraction-limited focal-plane blur, which is the real
                    # physical floor.
                    cm_for_min = getattr(self.blur_calc, 'calibration_model', None)
                    if cm_for_min is not None:
                        min_sigma_calib = float(cm_for_min.forward(0.0))
                    else:
                        min_sigma_calib = sigma_0
                    if self.min_blur_px is not None and self.min_blur_px > min_sigma_calib:
                        min_sigma_calib = self.min_blur_px
                    self.min_sigma_model = (min_sigma_calib
                                            * min_scale_ratio
                                            * self.native_to_model_scale)

                    # Log the full derivation so it's auditable. Method-aware
                    # formula display so quadrature/hybrid users see what
                    # actually got computed (not the linear formula).
                    cm_for_log = getattr(self.blur_calc, 'calibration_model', None)
                    if cm_for_log is None or cm_for_log.method == 'linear':
                        method_formula = (
                            f"    σ_calib = ρ × |z_max| + σ₀\n"
                            f"           = {rho:.4f} × {max_defocus:.1f} + {sigma_0:.4f}")
                    elif cm_for_log.method == 'quadrature':
                        sigma_floor_log = float(cm_for_log.sigma_floor_calib_px or 0.0)
                        method_formula = (
                            f"    σ_calib = sqrt((ρ × |z_max|)² + σ_floor²)\n"
                            f"           = sqrt(({rho:.4f} × {max_defocus:.1f})² "
                            f"+ {sigma_floor_log:.4f}²)")
                    else:  # hybrid
                        sigma_floor_log = float(cm_for_log.sigma_floor_calib_px or 0.0)
                        lut_n = (len(cm_for_log.residual_lut_mm_px)
                                  if cm_for_log.residual_lut_mm_px else 0)
                        method_formula = (
                            f"    σ_calib = sqrt((ρ × |z_max|)² + σ_floor²) + "
                            f"Δσ_lut(|z_max|)\n"
                            f"           [hybrid; LUT n={lut_n}]")
                    log(f"\nmax_sigma computation (direct mode, "
                        f"method={cm_for_log.method if cm_for_log else 'linear'}):")
                    log(f"  Step 1 — Calibration blur at max defocus:")
                    log(method_formula)
                    log(f"           = {self.max_sigma_calib:.4f} px  (at {scale_calib:.1f} px/mm)")
                    log(f"  Step 2 — Cross-camera correction (calib → training crop):")
                    log(f"    σ_crop  = σ_calib × (s_train / s_calib)")
                    sigma_crop = self.max_sigma_calib * scale_ratio
                    log(f"           = {self.max_sigma_calib:.4f} × ({max_training_scale:.1f} / {scale_calib:.1f})")
                    log(f"           = {sigma_crop:.4f} px  (at {max_training_scale:.1f} px/mm)")
                    log(f"  Step 3 — Resize crop to model input:")
                    log(f"    σ_model = σ_crop × (w_model / w_crop)")
                    log(f"           = {sigma_crop:.4f} × ({self.image_size} / {self.crop_size})")
                    log(f"           = {self.max_sigma:.4f} px  (at {self.image_size}×{self.image_size})")
                    log(f"  → max_sigma = {self.max_sigma:.4f} px")
                    log(f"  → min_sigma = {self.min_sigma_model:.4f} px  "
                        f"(from min scale {min_training_scale:.1f} px/mm, "
                        f"min_σ_calib = {min_sigma_calib:.4f} px)")
                    log(f"  Sampling uniform in model space: "
                        f"[{self.min_sigma_model:.4f}, {self.max_sigma:.4f}] px\n")
                else:
                    log("Warning: no valid scales found for training crops — "
                        "using provisional max_sigma (calibration scale only)")

            # Filter native_blur_map to training files only, so synthetic samples
            # draw baseline blur from the correct camera distribution.
            if native_blur_map and real_sharps:
                training_filenames = {p.name for p in real_sharps}
                filtered_native_blur = {
                    fn: v for fn, v in native_blur_map.items()
                    if fn in training_filenames
                }
                if filtered_native_blur:
                    native_blur_map = filtered_native_blur
                    log(f"Filtered native_blur_map to {len(native_blur_map)} training crops")

            # Extract appearance statistics from real crops for realistic synthetic generation.
            # Only needed when supplementing with synthetic images (< 50 real crops).
            if real_sharps and len(real_sharps) < 50:
                self.sphere_stats = SphereAppearanceStats.from_real_crops(
                    real_sharps, diameter_map=diameter_map,
                )

            # Create diameter bins if we have diameter info
            if diameter_map:
                # Match image paths to diameters
                images_with_diameters = []
                for img_path in real_sharps:
                    if img_path.name in diameter_map:
                        images_with_diameters.append((img_path, diameter_map[img_path.name]))

                if images_with_diameters:
                    # Calculate tertiles (33rd and 67th percentile) for 3 bins
                    diameters_only = [d for _, d in images_with_diameters]
                    p33 = np.percentile(diameters_only, 33.33)
                    p67 = np.percentile(diameters_only, 66.67)

                    # Store bin boundaries for final summary
                    bin_boundaries = (p33, p67)

                    # Create 3 bins: small, medium, large
                    diameter_bins = {0: [], 1: [], 2: []}
                    for img_path, diam in images_with_diameters:
                        if diam < p33:
                            diameter_bins[0].append((img_path, diam))
                        elif diam < p67:
                            diameter_bins[1].append((img_path, diam))
                        else:
                            diameter_bins[2].append((img_path, diam))

                    # Calculate median diameter for each bin (for synthetic generation)
                    bin_medians = [
                        np.median([d for _, d in diameter_bins[0]])
                        if diameter_bins[0] else np.median(diameters_only), np.median(
                            [d for _, d in diameter_bins[1]])
                        if diameter_bins[1] else np.median(diameters_only), np.median(
                            [d for _, d in diameter_bins[2]])
                        if diameter_bins[2] else np.median(diameters_only)]

                    use_binning = True
                    total_images = sum(len(diameter_bins[i]) for i in range(3))

                    log(f"Created 3 diameter bins (tertiles):")
                    log(
                        f"  Bin 0 (Small):  {len(diameter_bins[0])} images, median={bin_medians[0]:.1f}px")
                    log(
                        f"  Bin 1 (Medium): {len(diameter_bins[1])} images, median={bin_medians[1]:.1f}px")
                    log(
                        f"  Bin 2 (Large):  {len(diameter_bins[2])} images, median={bin_medians[2]:.1f}px")

                    if total_images >= 50:
                        log("Primary pool: 100% real templates from sharp crops "
                            "(uniform diameter distribution)")
                    else:
                        log(
                            f"Primary pool: supplementing bins with rendered "
                            f"droplets (only {total_images} real images)")

            # Fallback: no diameter info, use old behavior
            if not use_binning:
                log("Primary pool: no diameter binning - using random sampling")
                log(
                    f"Synthetic droplet diameter range: {diameter_range_px[0]}-{diameter_range_px[1]} px")
                if len(real_sharps) >= 50:
                    log("Using 100% real images (enough available)")
                else:
                    log(f"Using 70% real / 30% synthetic (only {len(real_sharps)} images)")

        metadata = []
        sample_idx = 0  # Track actual saved samples
        _sharp_cache = {}  # Cache decoded sharp images to avoid redundant disk reads

        # Cache primary sphere stats so we can swap to secondary_stats per
        # secondary sample and back. self.sphere_stats was set above (line
        # ~1399) when needed for rendering fallbacks.
        primary_sphere_stats = self.sphere_stats

        log(f"Starting generation of {num_samples} samples...")
        if sec_frac > 0 and secondary_sharps:
            primary_pct = (1.0 - sec_frac - ter_frac) * 100
            sec_pct = sec_frac * 100
            ter_pct = ter_frac * 100
            log(
                f"Three-source mixing enabled:\n"
                f"  Primary pool   (sharp crops + Gaussian):           "
                f"~{int(primary_pct)}% of {num_samples:,} = "
                f"{int(num_samples * (1 - sec_frac - ter_frac)):,} samples\n"
                f"  Secondary pool (calibration-rendered, donor-only): "
                f"~{int(sec_pct)}% of {num_samples:,} = "
                f"{int(num_samples * sec_frac):,} samples\n"
                f"  Tertiary pool  (real calib PNGs + ERF labels):     "
                f"~{int(ter_pct)}% of {num_samples:,} = "
                f"{int(num_samples * ter_frac):,} samples"
            )
        pbar = tqdm(total=num_samples, desc="Generating synthetic data")
        attempt = 0
        max_attempts = num_samples * 2  # Prevent infinite loop

        while sample_idx < num_samples and attempt < max_attempts:
            attempt += 1
            sharp_path = None  # Reset each iteration; set below if a real image is used
            use_secondary = False
            use_tertiary = False

            # ── Per-sample three-source dispatch ─────────────────────
            # Roll a random number, partition into three regions:
            #   [0, sec_frac)                   → secondary
            #   [sec_frac, sec_frac+ter_frac)   → tertiary
            #   [..., 1)                        → primary
            roll = random.random()
            if sec_frac > 0 and secondary_sharps and roll < sec_frac:
                use_secondary = True
            elif (ter_frac > 0 and tertiary_pool
                    and roll < (sec_frac + ter_frac)):
                use_tertiary = True

            if use_tertiary:
                # TERTIARY: bypass the synthesis pipeline entirely. Load a
                # real calibration PNG, augment lightly, scale label to
                # model space, write the sample directly without going
                # through generate_sample().
                self._save_tertiary_sample(
                    tertiary_pool, sample_idx, output_dir, _sharp_cache,
                    metadata, save_blur_trace, secondary_scale_px_per_mm,
                )
                sample_idx += 1
                pbar.update(1)
                progress_interval = max(1, num_samples // 10)
                if sample_idx % progress_interval == 0:
                    log(f"Progress: {sample_idx}/{num_samples} "
                        f"({100*sample_idx//num_samples}%)")
                continue

            if use_secondary:
                self.synthesis_size = secondary_synthesis_size
                if secondary_stats is not None:
                    self.sphere_stats = secondary_stats
                # Empty pool for the random-sampling branch -> always
                # falls through to the rendered-from-stats path. Each
                # secondary sample is a freshly-rendered sphere using
                # secondary_stats's appearance characteristics, then
                # Gaussian-blurred to a known sigma. Labels are exact
                # because the rendered template is truly sharp.
                use_binning_this_sample = False
                active_sharps = []
                source_label = 'secondary'
            else:
                self.synthesis_size = primary_synthesis_size
                self.sphere_stats = primary_sphere_stats
                use_binning_this_sample = use_binning
                active_sharps = real_sharps
                source_label = 'primary'

            # Decide which sharp image to use
            if use_binning_this_sample:
                # NEW: Binned sampling for uniform diameter distribution
                total_images = sum(len(diameter_bins[i]) for i in range(3))
                use_real_only = total_images >= 50

                # Cycle through bins uniformly (0, 1, 2, 0, 1, 2, ...)
                bin_idx = sample_idx % 3

                # Try to use real image from this bin
                if diameter_bins[bin_idx] and (use_real_only or random.random() < 0.7):
                    # Pick random image from this bin
                    sharp_path, actual_diameter = random.choice(diameter_bins[bin_idx])
                    _sp_key = str(sharp_path)
                    if _sp_key in _sharp_cache:
                        sharp = _sharp_cache[_sp_key].copy()
                    else:
                        sharp = cv2.imread(_sp_key, cv2.IMREAD_GRAYSCALE)
                        if sharp is not None:
                            _sharp_cache[_sp_key] = sharp.copy()

                    if sharp is None:
                        # Failed to load - fall back to synthetic at model scale
                        diameter_px = int(bin_medians[bin_idx] * self.image_size / self.crop_size)
                        sharp = None
                    else:
                        # Normalise to [0, 1]
                        sharp = sharp.astype(np.float32) / 255.0
                        sharp = self._prepare_image(sharp)
                        diameter_px = int(actual_diameter)  # Store actual diameter for metadata
                else:
                    # Generate synthetic droplet with bin median scaled to model resolution
                    diameter_px = int(bin_medians[bin_idx] * self.image_size / self.crop_size)
                    sharp = None
                    sharp_path = None
            else:
                # Random sampling (fallback when no diameter binning).
                # Uses active_sharps which is the per-sample chosen pool
                # (primary or secondary) — see two-source dispatch above.
                # If synthetic_fraction is explicitly set, use it; otherwise
                # fall back to the historical heuristic (real if >= 50, else
                # 70% real / 30% synthetic).
                if synthetic_fraction is not None:
                    sf = float(synthetic_fraction)
                    use_real_choice = (active_sharps
                                       and random.random() >= sf)
                else:
                    use_real_default = (len(active_sharps) >= 50
                                         if active_sharps else False)
                    use_real_choice = (active_sharps
                                       and (use_real_default
                                            or random.random() < 0.7))

                if use_real_choice:
                    # Load random real sharp image from the active pool
                    sharp_path = random.choice(active_sharps)
                    _sp_key = str(sharp_path)
                    if _sp_key in _sharp_cache:
                        sharp = _sharp_cache[_sp_key].copy()
                    else:
                        sharp = cv2.imread(_sp_key, cv2.IMREAD_GRAYSCALE)
                        if sharp is not None:
                            _sharp_cache[_sp_key] = sharp.copy()

                    if sharp is None:
                        # Failed to load - fall back to synthetic
                        diameter_px = None  # Let generate_sample pick from sphere_stats
                        sharp = None
                        sharp_path = None
                    else:
                        # Normalise to [0, 1]
                        sharp = sharp.astype(np.float32) / 255.0
                        sharp = self._prepare_image(sharp)
                        diameter_px = None  # Unknown for real images
                else:
                    # Generate synthetic droplet (uses self.sphere_stats which
                    # was swapped to secondary_stats above if applicable)
                    diameter_px = None  # Let generate_sample pick from sphere_stats
                    sharp = None
                    sharp_path = None

            # Look up per-crop values for this image
            if sharp_path is not None:
                scale_camera = scale_map.get(sharp_path.name)
                native_blur = native_blur_map.get(sharp_path.name, 0.0)
            elif use_secondary:
                # Secondary calibration-rendered sample: use the
                # calibration camera's scale (from the bundle's yaml)
                # and zero native blur (rendered spheres are truly sharp).
                scale_camera = secondary_scale_px_per_mm
                native_blur = 0.0
            else:
                # Primary synthetic sample: pick a donor crop and take both
                # native_blur and scale from it so cross-camera correction
                # is consistent.
                if native_blur_map:
                    donor_fn = random.choice(list(native_blur_map.keys()))
                    native_blur = native_blur_map[donor_fn]
                    scale_camera = scale_map.get(donor_fn)
                else:
                    native_blur = 0.0
                    scale_camera = None

            # Generate sample
            sample = self.generate_sample(
                sharp_image=sharp,
                diameter_px=diameter_px,
                scale_px_per_mm=scale_camera,
                native_blur_sigma=native_blur,
            )

            # Add noise if requested
            if add_noise and noise_level > 0:
                noise = np.random.normal(0, noise_level, sample['blur'].shape)
                sample['blur'] = np.clip(sample['blur'] + noise, 0, 1)

            # Save images with sequential index (no gaps)
            idx_str = f"{sample_idx:06d}"

            cv2.imwrite(
                str(output_dir / 'sharp' / f'{idx_str}.png'),
                (sample['sharp'] * 255).astype(np.uint8)
            )
            cv2.imwrite(
                str(output_dir / 'blur' / f'{idx_str}.png'),
                (sample['blur'] * 255).astype(np.uint8)
            )
            # Mode-aware map key: sigma_map for direct, coc_map for optical
            map_key = 'sigma_map' if self.params.training_mode == 'direct' else 'coc_map'
            cv2.imwrite(
                str(output_dir / 'blur_map' / f'{idx_str}.png'),
                (sample[map_key] * 255).astype(np.uint8)
            )

            # Store metadata with mode-specific column name
            blur_key = 'sigma_px' if self.params.training_mode == 'direct' else 'coc_px'
            value_key = 'sigma_value' if self.params.training_mode == 'direct' else 'coc_value'
            _diam_model = (round(diameter_px * self.native_to_model_scale)
                           if diameter_px is not None else '')
            # Phase 5: store abs_z_mm alongside legacy signed defocus_mm.
            # source_z_signed_mm is empty for primary/secondary (the sign
            # in defocus_mm is randomly assigned, not physical truth —
            # only tertiary samples have meaningful signed z).
            row = {
                'index': idx_str,
                blur_key: sample[value_key],
                'defocus_mm': sample['defocus_mm'],
                'abs_z_mm': abs(sample['defocus_mm']),
                'source_z_signed_mm': '',
                'diameter_px': diameter_px if diameter_px is not None else '',
                'diameter_model_px': _diam_model,
            }

            if save_blur_trace and self.params.training_mode == 'direct':
                # Tag source: real sharp / calibration-rendered / fallback synthetic
                if sharp_path is not None:
                    _src = sharp_path.name
                    _cam = camera_map.get(sharp_path.name, '')
                elif use_secondary:
                    _src = 'calib_render'
                    _cam = 'calibration'
                else:
                    _src = 'synthetic'
                    _cam = ''
                _scale = scale_camera
                _scale_calib = self.params.scale_calib_px_per_mm
                _cc = (_scale / _scale_calib
                       if (_scale is not None and _scale_calib is not None and _scale_calib > 0)
                       else None)
                _rho = self.params.rho_direct
                _abs_def = abs(sample['defocus_mm'])
                _sigma_0 = self.params.sigma_0 if self.params.sigma_0 is not None else 0.0
                _sig_cal = (_rho * _abs_def + _sigma_0) if _rho else 0.0
                _sig_nat_exp = _sig_cal * _cc if _cc is not None else _sig_cal
                _sig_mdl_exp = _sig_nat_exp * self.native_to_model_scale
                _sig_applied = sample.get('sigma_kernel', float('nan'))
                _sig_err = _sig_applied - _sig_mdl_exp
                _sig_err_pct = (100.0 * _sig_err / _sig_mdl_exp
                                if _sig_mdl_exp != 0 else float('nan'))
                _native_blur_model = native_blur * self.native_to_model_scale
                row.update({
                    'source_image': _src,
                    'camera': _cam,
                    'scale_px_per_mm': '' if _scale is None else _scale,
                    'scale_calib_px_per_mm': '' if _scale_calib is None else _scale_calib,
                    'cc_factor': '' if _cc is None else round(_cc, 6),
                    'rho_direct': _rho,
                    'sigma_calib_px': round(_sig_cal, 6),
                    'sigma_native_expected_px': round(_sig_nat_exp, 6),
                    'sigma_model_expected_px': round(_sig_mdl_exp, 6),
                    'sigma_applied_px': round(_sig_applied, 6),
                    'native_blur_crop_px': round(native_blur, 6),
                    'native_blur_model_px': round(_native_blur_model, 6),
                    'quadrature_error_px': round(_sig_err, 6),
                    'quadrature_error_pct': round(_sig_err_pct, 4),
                    'crop_size_px': self.crop_size,
                    'model_size_px': self.image_size,
                })

            # ERF validation for selected samples
            if erf_validation and sample_idx in validate_indices:
                erf_result = validate_sample_erf(
                    sample['blur'],
                    sigma_label=sample[value_key],
                )
                row.update(erf_result)

            metadata.append(row)

            sample_idx += 1
            pbar.update(1)

            # Log progress every 10%
            progress_interval = max(1, num_samples // 10)
            if sample_idx % progress_interval == 0:
                log(f"Progress: {sample_idx}/{num_samples} ({100*sample_idx//num_samples}%)")

        pbar.close()

        import pandas as pd
        df = pd.DataFrame(metadata)
        # Set the image number as the actual index
        df = df.set_index('index')
        df.to_csv(output_dir / 'metadata.csv', index=True)

        log(f"Generated {len(metadata)} samples in {output_dir}")
        log("Note: All configuration is stored in training_config.yaml")

        # Print diameter category distribution
        if 'diameter_px' in df.columns:
            # Coerce to numeric — secondary/tertiary samples write '' for
            # diameter_px (their effective diameter is internal); drop those.
            diameter_values = pd.to_numeric(
                df['diameter_px'], errors='coerce').dropna().values

            if (use_binning and bin_boundaries is not None
                    and len(diameter_values) > 0):
                # Use actual tertile boundaries from stratified sampling
                p33, p67 = bin_boundaries
                min_diam = diameter_values.min()
                max_diam = diameter_values.max()

                small = np.sum(diameter_values < p33)
                medium = np.sum((diameter_values >= p33) & (diameter_values < p67))
                large = np.sum(diameter_values >= p67)

                total_with_diam = len(diameter_values)

                log("\nDiameter Category Distribution (Primary pool only — "
                    "secondary/tertiary samples don't have droplet diameters):")
                log(f"  Small  ({min_diam:.0f}-{p33:.0f} px):   {small:5d} ({small/total_with_diam*100:5.1f}%)")
                log(f"  Medium ({p33:.0f}-{p67:.0f} px):  {medium:5d} ({medium/total_with_diam*100:5.1f}%)")
                log(f"  Large  ({p67:.0f}-{max_diam:.0f} px): {large:5d} ({large/total_with_diam*100:5.1f}%)")
                log(f"  Total with diameter: {total_with_diam}  (out of {len(df)} total samples)")
                log("  Note: Bins sampled uniformly for balanced distribution")
            else:
                # Fallback: show basic statistics without stratified info
                min_diam = diameter_values.min()
                max_diam = diameter_values.max()
                mean_diam = diameter_values.mean()

                log("\nDiameter Distribution (Random Sampling):")
                log(f"  Range: {min_diam:.0f} - {max_diam:.0f} px")
                log(f"  Mean: {mean_diam:.1f} px")
                log(f"  Total: {len(diameter_values)}")
                log("  Note: No stratified sampling (weighted by input availability)")

        # Write dataset_summary.json — small machine+human readable record
        import json
        from datetime import datetime
        blur_col = 'sigma_px' if 'sigma_px' in df.columns else 'coc_px'
        summary = {
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'n_samples': int(len(df)),
            'image_size_px': int(self.image_size),
            'training_mode': self.params.training_mode,
            'blur_column': blur_col,
            'blur_range_px': [float(df[blur_col].min()), float(df[blur_col].max())],
            'distribution': self.blur_distribution,
            'beta_alpha': float(self.beta_alpha) if self.beta_alpha is not None else None,
            'beta_beta': float(self.beta_beta) if self.beta_beta is not None else None,
            'min_blur_px': float(self.min_blur_px) if self.min_blur_px is not None else None,
            'defocus_range_mm': [float(self.defocus_range[0]), float(self.defocus_range[1])],
            'rho_direct': float(self.params.rho_direct) if self.params.rho_direct else None,
            'sigma_0': float(self.params.sigma_0) if self.params.sigma_0 is not None else None,
            'scale_calib_px_per_mm': (
                float(self.params.scale_calib_px_per_mm)
                if self.params.scale_calib_px_per_mm else None
            ),
            'diameter_bins_used': bool(use_binning),
        }
        if 'diameter_px' in df.columns:
            # Coerce to numeric — secondary (rendered) samples write ''
            # for diameter_px (the rendered diameter is internal to
            # generate_sample); coerce drops them to NaN before min/max.
            _diam = pd.to_numeric(df['diameter_px'], errors='coerce').dropna()
            if not _diam.empty:
                summary['diameter_range_px'] = [
                    float(_diam.min()), float(_diam.max())
                ]
        with open(output_dir / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Phase 5: persist the unified CalibrationModel as its own
        # `calibration_model.yaml` next to generation_config.yaml. This is
        # the source of truth for downstream stages (training reads it,
        # bakes into checkpoint; inference reads from checkpoint).
        # When missing, training falls back to building a linear model
        # from rho_direct/sigma_0 in generation_config.yaml (back-compat).
        cm = getattr(self, 'calibration_model', None)
        if cm is not None:
            try:
                import yaml as _yaml
                cm_doc = {
                    'calibration_model': cm.to_dict(),
                    'sha256': cm.sha256(),
                    'method': cm.method,
                }
                with open(output_dir / 'calibration_model.yaml', 'w') as f:
                    _yaml.dump(cm_doc, f, default_flow_style=False, sort_keys=False)
                log(f"  Wrote calibration_model.yaml (method={cm.method}, sha256={cm.sha256()[:12]}...)")
            except Exception as e:
                log(f"  WARNING: could not write calibration_model.yaml: {e}")

        return {
            'diameter_bins_used': use_binning,
            'diameter_bin_boundaries': bin_boundaries if use_binning else None
        }

    def _save_tertiary_sample(
        self,
        tertiary_pool: list,
        sample_idx: int,
        output_dir: Path,
        sharp_cache: dict,
        metadata: list,
        save_blur_trace: bool,
        calib_scale_px_per_mm: Optional[float],
    ) -> None:
        """Save one TERTIARY sample (real calibration PNG + ERF label).

        Bypasses the synthesis pipeline. The PNG is the blur, the ERF
        sigma from measurements.csv is the label. Light augmentation
        (small position offset, brightness shift, optional flip)
        prevents memorisation of the ~61 unique calibration frames.
        """
        png_path, source_size, sigma_at_source = random.choice(tertiary_pool)
        cache_key = f"_ter_{png_path}"
        if cache_key in sharp_cache:
            img = sharp_cache[cache_key].copy()
        else:
            img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Fallback: skip and let outer loop re-roll
                return
            sharp_cache[cache_key] = img.copy()

        # Light augmentation (preserves sigma)
        # Random small position offset (±2% of width)
        h, w = img.shape[:2]
        max_off = max(1, int(min(h, w) * 0.02))
        off_y = random.randint(-max_off, max_off)
        off_x = random.randint(-max_off, max_off)
        if off_y or off_x:
            M = np.float32([[1, 0, off_x], [0, 1, off_y]])
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # Brightness ±5%
        bright = 1.0 + random.uniform(-0.05, 0.05)
        img_f = np.clip(img.astype(np.float32) * bright, 0, 255)
        # Optional horizontal flip (sphere is rotationally symmetric → label unchanged)
        if random.random() < 0.5:
            img_f = img_f[:, ::-1].copy()

        # Resize to model size
        ms = self.image_size
        if max(img_f.shape) != ms:
            interp = cv2.INTER_AREA if max(img_f.shape) > ms else cv2.INTER_CUBIC
            img_f = cv2.resize(img_f, (ms, ms), interpolation=interp)
        img_uint8 = img_f.astype(np.uint8)

        # Scale ERF sigma from source-px to model-px
        sigma_model = sigma_at_source * (ms / source_size)

        # Save image
        idx_str = f"{sample_idx:06d}"
        cv2.imwrite(str(output_dir / 'sharp' / f'{idx_str}.png'), img_uint8)
        cv2.imwrite(str(output_dir / 'blur' / f'{idx_str}.png'), img_uint8)
        # Blur map: constant normalised sigma
        norm = min(sigma_model / self.max_sigma, 1.0) if self.max_sigma > 0 else 0.0
        sigma_map_img = np.full_like(img_uint8, int(round(norm * 255)))
        cv2.imwrite(str(output_dir / 'blur_map' / f'{idx_str}.png'), sigma_map_img)

        # Build metadata row
        blur_key = 'sigma_px' if self.params.training_mode == 'direct' else 'coc_px'
        # Recover defocus from sigma using calibration.
        # Phase 5: prefer the CalibrationModel when set (handles
        # quadrature/hybrid correctly — fixes bug #2: tertiary frames at
        # true z=0 were getting fictional ±0.63 defocus from the linear
        # formula because σ_0 (linear y-intercept) ≈ 0.35 underestimates
        # the true focal-plane blur ≈ 1.0 px).
        rho = self.params.rho_direct
        sigma_0 = self.params.sigma_0 if self.params.sigma_0 is not None else 0.0
        cm = getattr(self, 'calibration_model', None)
        if cm is not None:
            sigma_calib = sigma_at_source  # already in calibration source-px
            abs_def, _flag = cm.inverse(sigma_calib)
            if abs_def != abs_def:  # nan-safe (SATURATED case)
                abs_def = 0.0
            defocus_mm = abs_def * (1 if random.random() < 0.5 else -1)
        elif rho and rho > 0 and calib_scale_px_per_mm:
            sigma_calib = sigma_at_source  # already in calibration source-px
            abs_def = max((sigma_calib - sigma_0) / rho, 0.0)
            # Sign: random ± since the sphere image alone doesn't disambiguate
            defocus_mm = abs_def * (1 if random.random() < 0.5 else -1)
        else:
            abs_def = 0.0
            defocus_mm = 0.0
        # Phase 5: extract the REAL signed z from the calibration PNG
        # filename (e.g. `9mm_z-3.5mm.png` → −3.5). Stored as
        # `source_z_signed_mm` for diagnostic / asymmetry analysis.
        # Primary/secondary samples leave this column empty.
        try:
            from run_paths import parse_true_z_from_filename
            source_z_signed = parse_true_z_from_filename(png_path.name)
        except Exception:
            source_z_signed = None
        row = {
            'index': idx_str,
            blur_key: sigma_model,
            'defocus_mm': defocus_mm,
            # Phase 5: new honest columns alongside legacy defocus_mm
            'abs_z_mm': abs_def,
            'source_z_signed_mm': (source_z_signed
                                    if source_z_signed is not None else ''),
            'diameter_px': '',
            'diameter_model_px': '',
        }
        if save_blur_trace and self.params.training_mode == 'direct':
            row.update({
                'source_image': f'calib_real_{png_path.name}',
                'camera': 'calibration',
                'scale_px_per_mm': (calib_scale_px_per_mm
                                    if calib_scale_px_per_mm else ''),
                'scale_calib_px_per_mm': (
                    self.params.scale_calib_px_per_mm
                    if self.params.scale_calib_px_per_mm else ''),
                'cc_factor': 1.0,
                'rho_direct': rho,
                'sigma_calib_px': round(sigma_at_source, 6),
                'sigma_native_expected_px': round(sigma_at_source, 6),
                'sigma_model_expected_px': round(sigma_model, 6),
                'sigma_applied_px': round(sigma_model, 6),
                'native_blur_crop_px': round(sigma_at_source, 6),
                'native_blur_model_px': round(sigma_model, 6),
                'quadrature_error_px': 0.0,
                'quadrature_error_pct': 0.0,
                'crop_size_px': source_size,
                'model_size_px': self.image_size,
            })
        metadata.append(row)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size (preserves droplet, scales down).

        If self.synthesis_size is set, that's used as the intermediate
        target so blur is applied at higher resolution. The sample is
        downsampled to self.image_size at the end of generate_sample.
        """
        h, w = image.shape[:2]
        target = self.synthesis_size if self.synthesis_size else self.image_size

        # If image is larger, resize it down (don't just crop!)
        if h > target or w > target:
            # Calculate scale to fit within target
            scale = target / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)

            # Resize using area interpolation (best for downscaling)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]
        # If image is smaller AND synthesis_size > image_size, upscale
        # so blur can be applied at the chosen synthesis resolution.
        elif self.synthesis_size and (h < target or w < target):
            scale = target / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = image.shape[:2]

        # Pad to exact target size if needed (centers the image)
        if h < target or w < target:
            pad_h = (target - h) // 2
            pad_w = (target - w) // 2
            image = np.pad(
                image,
                ((pad_h, target - h - pad_h), (pad_w, target - w - pad_w)),
                mode='constant',
                constant_values=1.0  # White background
            )

        return image


# =============================================================================
# CLI Interface
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic blurred training data"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='training_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--sharp-dir', '-s',
        type=str,
        default=None,
        help='Directory containing sharp droplet crops'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/synthetic',
        help='Output directory for generated data'
    )
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=50000,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=128,
        help='Output image size (model size, square)'
    )
    parser.add_argument(
        '--crop-size',
        type=int,
        default=None,
        help='Native crop size before resize (for resolution scaling). Defaults to image-size if not set.'
    )
    parser.add_argument(
        '--add-noise',
        action='store_true',
        help='Add Gaussian noise to blurred images'
    )
    parser.add_argument(
        '--noise-level',
        type=float,
        default=0.01,
        help='Noise standard deviation'
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    optical_params = BlurParams.from_config(config)

    data_config = config.get('data', {})
    defocus_range = tuple(data_config.get('defocus_range_mm', [-12.0, 12.0]))
    diameter_range = tuple(data_config.get('droplet_diameter_range_px', [10, 50]))

    generator = SyntheticBlurGenerator(
        optical_params=optical_params,
        defocus_range_mm=defocus_range,
        image_size=args.image_size,
        crop_size=args.crop_size  # None defaults to image_size in the class
    )

    generator.generate_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        sharp_images_dir=args.sharp_dir,
        diameter_range_px=diameter_range,
        add_noise=args.add_noise,
        noise_level=args.noise_level
    )


if __name__ == "__main__":
    main()
