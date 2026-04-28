"""
Inference Engine for Defocus Estimation from Shadowgraphy Recordings.

Loads a .cine file, selects the best pre-collision frame, crops around the
droplet, applies boundary normalisation, and runs the trained CNN to produce
a defocus displacement in mm.

All heavy logic lives here; the GUI calls into this module.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path manipulation so we can import from preprocessing and training modules
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

try:
    # Package-style imports (when installed via pip install -e .)
    from Preprocessing.cine_io import safe_load_cine, PYPHANTOM_AVAILABLE
    from Preprocessing.darkness_analysis import choose_best_frame_geometry_only
    from Preprocessing.cropping import crop_droplet_with_sphere_guard
    from Training.model import DefocusNet
    from physics import ScalingParams, invert_prediction, defocus_uncertainty
except ImportError:
    # Fallback: add sibling directories to path
    for _p in (_REPO_ROOT / "Preprocessing", _REPO_ROOT / "Training"):
        _ps = str(_p)
        if _ps not in sys.path:
            sys.path.insert(0, _ps)
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from cine_io import safe_load_cine, PYPHANTOM_AVAILABLE          # noqa: E402
    from darkness_analysis import choose_best_frame_geometry_only     # noqa: E402
    from cropping import crop_droplet_with_sphere_guard               # noqa: E402
    from model import DefocusNet                                      # noqa: E402
    from physics import ScalingParams, invert_prediction, defocus_uncertainty  # noqa: E402

# flatten_sphere_crop lives in Calibration/sphere_processing.py
try:
    from Calibration.sphere_processing import flatten_sphere_crop
except ImportError:
    _calib_dir = _REPO_ROOT / "Calibration"
    if str(_calib_dir) not in sys.path:
        sys.path.insert(0, str(_calib_dir))
    from sphere_processing import flatten_sphere_crop  # type: ignore  # noqa: E402


# ── Default calibration constants ──────────────────────────────────────────
DEFAULT_RHO = 1.548              # px/mm  (calibration slope)
DEFAULT_SIGMA_0 = 0.125          # px     (calibration intercept)
DEFAULT_S_CALIB = 102.57         # px/mm  (calibration camera scale)
DEFAULT_S_C = 102.57             # px/mm  (inference camera scale — same camera by default)
DEFAULT_FEATHER_PX = 40          # Gaussian feather width in pixels
DEFAULT_CROP_SIZE = 299          # Training crop size
MODEL_INPUT_SIZE = 256           # Network input resolution
DEFAULT_FLATTEN_MODE = "calibration"  # "calibration" (cfg1) or "boundary_normalise" (cfg4)
DEFAULT_INNER_MARGIN_PX = 20     # cfg1 inner margin — preserves blur up to this distance from edge


# ── Boundary Normalisation ─────────────────────────────────────────────────

def boundary_normalise(
    crop: np.ndarray,
    feather_px: int = DEFAULT_FEATHER_PX,
) -> np.ndarray:
    """
    Otsu threshold + Gaussian feather boundary normalisation.

    1. Convert to uint8 if needed.
    2. Otsu threshold to get a binary mask of the droplet (dark object).
    3. Compute signed distance transform (positive inside, negative outside).
    4. Apply cosine feather at the boundary: interior fades to 0, exterior to 1.

    Returns a float32 image in [0, 1].
    """
    if crop.dtype != np.uint8:
        if crop.max() <= 1.0:
            img_u8 = (crop * 255).astype(np.uint8)
        else:
            img_u8 = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_u8 = crop

    # Otsu threshold — dark droplet on bright background
    _, binary = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_mask = (binary == 0).astype(np.uint8)  # 1 where droplet is

    # Fill holes in the mask — caustics (bright spots inside the droplet)
    # appear as holes in the dark mask and must be filled so the interior
    # is uniformly suppressed
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dark_mask, contours, -1, 1, cv2.FILLED)

    # Signed distance: positive inside droplet, negative outside
    dist_inside = cv2.distanceTransform(dark_mask, cv2.DIST_L2, 5).astype(np.float32)
    dist_outside = cv2.distanceTransform(1 - dark_mask, cv2.DIST_L2, 5).astype(np.float32)
    signed_dist = np.where(dark_mask > 0, dist_inside, -dist_outside)

    img_f = img_u8.astype(np.float32) / 255.0
    out = img_f.copy()
    fw = max(feather_px, 1)

    # Deep interior → 0 (black)
    out[signed_dist > fw] = 0.0

    # Interior feather zone: cosine fade from original to 0
    mask_inner = (signed_dist > 0) & (signed_dist <= fw)
    if np.any(mask_inner):
        t = np.clip(signed_dist[mask_inner] / fw, 0, 1)
        out[mask_inner] = 0.5 * (1 + np.cos(np.pi * t)) * img_f[mask_inner]

    # Exterior feather zone: cosine fade from original to 1
    mask_outer = (signed_dist < 0) & (signed_dist >= -fw)
    if np.any(mask_outer):
        t = np.clip(-signed_dist[mask_outer] / fw, 0, 1)
        out[mask_outer] = img_f[mask_outer] + 0.5 * (1 - np.cos(np.pi * t)
                                                     ) * (1.0 - img_f[mask_outer])

    # Deep exterior → 1 (white / background)
    out[signed_dist < -fw] = 1.0

    return out


# ── Inference Engine ───────────────────────────────────────────────────────

class InferenceEngine:
    """End-to-end pipeline: .cine → defocus displacement (mm)."""

    def __init__(self, settings: Dict[str, Any]):
        """Initialise engine with calibration and pipeline settings.

        Args:
            settings: Must contain at least 'model_path'. Optional keys:
                rho, sigma_0, s_calib, s_c, feather_px, crop_size, device.
        """
        self.settings = settings
        self.model: Optional[DefocusNet] = None
        self.max_blur: float = 20.0
        self.model_size: int = MODEL_INPUT_SIZE
        self.config: Dict[str, Any] = {}
        self.training_mode: str = "direct"
        self.scale_calib: Optional[float] = None

    # ── Model loading ──────────────────────────────────────────────────

    def load_model(self) -> str:
        """
        Load the CNN from the checkpoint specified in settings.

        Returns a status message string.
        """
        model_path = Path(self.settings.get("model_path", ""))
        if not model_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        device_str = self.settings.get("device", "cpu")
        self.device = torch.device(device_str)

        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=True)

        if "config" in checkpoint:
            self.config = checkpoint["config"]
        else:
            raise ValueError("Checkpoint does not contain a config dict.")

        self.training_mode = checkpoint.get("training_mode", "direct")

        self.model = DefocusNet.from_config(self.config).to(self.device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "dme_state_dict" in checkpoint:
            state = checkpoint["dme_state_dict"]
            sample_key = next(iter(state))
            if not sample_key.startswith("dme_subnet."):
                state = {f"dme_subnet.{k}": v for k, v in state.items()}
            self.model.load_state_dict(state)
        else:
            raise ValueError("Checkpoint missing model weights.")

        self.model.eval()

        ckpt_max = checkpoint.get("max_blur", checkpoint.get("max_coc"))
        if ckpt_max is not None:
            self.max_blur = float(ckpt_max)
        else:
            data_cfg = self.config.get("data", {})
            if "blur_range_px" in data_cfg:
                self.max_blur = float(data_cfg["blur_range_px"][1])
            else:
                self.max_blur = 20.0

        data_cfg = self.config.get("data", {})
        self.model_size = int(data_cfg.get("image_size_px", MODEL_INPUT_SIZE))

        training_cfg = self.config.get("training", {})
        self.scale_calib = training_cfg.get("scale_calib_px_per_mm")

        # Phase 7: build the unified CalibrationModel from checkpoint config
        # if present. Falls back to building a linear model from
        # rho_direct/sigma_0 (legacy checkpoints).
        self.calibration_model = self._build_calibration_model_from_config()

        user_crop = int(self.settings.get("crop_size", DEFAULT_CROP_SIZE))
        mismatch_msg = ""
        if user_crop != self.model_size:
            mismatch_msg = (
                f"  |  WARNING: crop_size={user_crop} != "
                f"model training size={self.model_size}"
            )

        return (
            f"Model loaded  |  mode={self.training_mode}  |  "
            f"max_blur={self.max_blur:.2f} px  |  "
            f"model_size={self.model_size}{mismatch_msg}"
        )

    # ── Frame selection ────────────────────────────────────────────────

    def select_best_frame(self, cine_path: Path) -> Tuple[Any, int, Dict[str, Any], np.ndarray]:
        """
        Open a .cine and find the best pre-collision frame.

        Returns (cine_obj, frame_idx, geometry_dict, frame_gray).
        """
        if not PYPHANTOM_AVAILABLE:
            raise RuntimeError(
                "pyphantom is not installed — cannot read .cine files."
            )

        cine_obj = safe_load_cine(cine_path)
        if cine_obj is None:
            raise RuntimeError(f"Failed to open: {cine_path.name}")

        frame_idx, geo = choose_best_frame_geometry_only(cine_obj)

        # Retrieve the grayscale frame for display / cropping
        from image_utils import load_frame_gray  # noqa: E402 (already on path)
        frame_gray = load_frame_gray(cine_obj, frame_idx)

        return cine_obj, frame_idx, geo, frame_gray

    # ── Crop extraction ────────────────────────────────────────────────

    def extract_crop(
        self,
        frame_gray: np.ndarray,
        geo: Dict[str, Any],
    ) -> np.ndarray:
        """
        Crop the droplet region from the frame, excluding the sphere.

        Uses crop_droplet_with_sphere_guard from the preprocessing module.
        """
        crop_size = int(self.settings.get("crop_size", DEFAULT_CROP_SIZE))

        y_top = geo.get("y_top")
        y_bottom = geo.get("y_bottom")
        cx = geo.get("cx")
        y_sphere = geo.get("y_bottom_sphere")

        if y_top is None or y_bottom is None or cx is None:
            # Fallback: centre crop
            h, w = frame_gray.shape
            y0 = max(0, h // 2 - crop_size // 2)
            x0 = max(0, w // 2 - crop_size // 2)
            return frame_gray[y0:y0 + crop_size, x0:x0 + crop_size]

        crop = crop_droplet_with_sphere_guard(
            frame=frame_gray,
            y_top=y_top,
            y_bottom=y_bottom,
            cx=cx,
            target_w=crop_size,
            target_h=crop_size,
            y_sphere=y_sphere,
            safety=3,
        )
        return crop

    # ── Preprocessing ──────────────────────────────────────────────────

    def preprocess_crop(self, crop: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Apply boundary normalisation, resize, and convert to tensor.

        Returns (normalised_image_for_display, model_tensor).
        normalised_image_for_display is float32 in [0, 1].
        model_tensor is (1, 1, 256, 256) float32 in [-1, 1].

        The flatten recipe is selected by ``settings['flatten_mode']``:
          - ``"calibration"`` (default): flatten_sphere_crop with
            inner_margin=20, flatten_exterior=False — matches what tertiary
            training samples were processed with. Empirically gives
            6-21x lower MAE than ``"boundary_normalise"`` on calibration
            spheres in the trustworthy regime.
          - ``"boundary_normalise"``: legacy Otsu + cosine feather.
        """
        feather_px = int(self.settings.get("feather_px", DEFAULT_FEATHER_PX))
        flatten_mode = str(self.settings.get("flatten_mode",
                                              DEFAULT_FLATTEN_MODE))
        if flatten_mode == "calibration":
            inner_margin = int(self.settings.get("inner_margin_px",
                                                  DEFAULT_INNER_MARGIN_PX))
            if crop.dtype == np.uint8:
                img_f = crop.astype(np.float32) / 255.0
            else:
                img_f = crop.astype(np.float32)
                if img_f.max() > 1.5:
                    img_f = img_f / 255.0
            flat, info = flatten_sphere_crop(img_f, inner_margin=inner_margin,
                                              flatten_exterior=False)
            if info is None:
                # sphere detection failed — fall back to legacy recipe
                norm_img = boundary_normalise(crop, feather_px=feather_px)
            else:
                norm_img = flat
        else:
            norm_img = boundary_normalise(crop, feather_px=feather_px)

        h, w = norm_img.shape[:2]
        if h > self.model_size or w > self.model_size:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_CUBIC
        resized = cv2.resize(norm_img, (self.model_size, self.model_size), interpolation=interp)

        tensor_input = resized.astype(np.float32) * 2.0 - 1.0
        tensor_input = torch.from_numpy(tensor_input).unsqueeze(0).unsqueeze(0)

        if self.model is not None:
            tensor_input = tensor_input.to(self.device)

        return norm_img, tensor_input

    def _build_calibration_model_from_config(self):
        """Phase 7: build a CalibrationModel from the checkpoint config.

        Three sources, in order of preference:
        1. ``training.calibration_model`` block (Phase 6 emit) — most info,
           includes method, all bounds, and sigma_max_model_observed.
        2. Legacy ``training.rho_direct`` / ``sigma_0`` — builds a linear
           model with no trust bounds (back-compat).
        3. Returns None if neither available (caller falls back to
           ScalingParams flow with no bounds_flag).
        """
        try:
            sys_path = sys.path
            from physics import CalibrationModel
        except Exception:
            try:
                from CROPPING.physics import CalibrationModel  # type: ignore
            except Exception:
                return None
        training_cfg = self.config.get("training", {}) if self.config else {}

        cm_dict = training_cfg.get("calibration_model")
        if isinstance(cm_dict, dict) and cm_dict:
            try:
                model = CalibrationModel.from_dict(cm_dict)
                method_label = training_cfg.get("inversion_method", model.method)
                print(f"  Loaded CalibrationModel from checkpoint "
                      f"(method={method_label}, sha256={model.sha256()[:12]}...)")
                return model
            except Exception as e:
                print(f"  WARNING: could not deserialise calibration_model "
                      f"from checkpoint ({e}); falling back to linear")

        rho = training_cfg.get("rho_direct")
        sigma_0 = training_cfg.get("sigma_0")
        if rho is None or sigma_0 is None:
            print("  Checkpoint has no inversion_method or rho_direct; "
                  "calibration model unavailable.")
            return None
        s_calib = training_cfg.get("scale_calib_px_per_mm")
        try:
            model = CalibrationModel.from_legacy_scaling(
                rho=float(rho), sigma_0=float(sigma_0),
                s_calib_px_per_mm=(float(s_calib) if s_calib is not None else None),
            )
            print(f"  Built fallback linear CalibrationModel from "
                  f"rho_direct={rho}, sigma_0={sigma_0} (no inversion_method "
                  f"in checkpoint — defaulting to linear)")
            return model
        except Exception as e:
            print(f"  Could not build CalibrationModel: {e}")
            return None

    # ── Model forward pass + inversion chain ───────────────────────────

    def run_inference(
        self,
        tensor_input: torch.Tensor,
        native_crop_size: int,
    ) -> Dict[str, float]:
        """Forward pass through model and inverse chain to defocus (mm).

        Args:
            tensor_input: (1, 1, 256, 256) tensor in [-1, 1].
            native_crop_size: dimension of the original crop before resize.

        Returns:
            Dict with pred_norm, sigma_model, sigma_native, defocus_mm,
            saturated/clamped flags, and provenance fields.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        with torch.no_grad():
            pred_norm = self.model(tensor_input)

        pred_val = pred_norm.squeeze().item()

        rho = float(self.settings.get("rho", DEFAULT_RHO))
        sigma_0 = float(self.settings.get("sigma_0", DEFAULT_SIGMA_0))
        s_calib = float(self.settings.get("s_calib", DEFAULT_S_CALIB))
        s_c = float(self.settings.get("s_c", DEFAULT_S_C))
        feather_px = int(self.settings.get("feather_px", DEFAULT_FEATHER_PX))
        crop_size = int(self.settings.get("crop_size", DEFAULT_CROP_SIZE))

        params = ScalingParams(
            rho=rho, sigma_0=sigma_0,
            s_calib=s_calib, s_inference=s_c,
            max_blur=self.max_blur, model_size=self.model_size,
        )

        result = invert_prediction(pred_val, params, native_crop_size)

        rho_std = float(self.settings.get("rho_std", 0.0))
        sigma_0_std = float(self.settings.get("sigma_0_std", 0.0))
        unc_mm = 0.0
        # Phase 10: prefer method-aware uncertainty from CalibrationModel
        # when its loo_cv field is populated. Falls back to legacy linear
        # propagation if not present (back-compat).
        cm_unc = getattr(self, 'calibration_model', None)
        if cm_unc is not None and getattr(cm_unc, 'loo_cv', None):
            try:
                unc_mm = cm_unc.defocus_uncertainty_at(
                    sigma_model_px=result.sigma_model,
                    s_inf_px_per_mm=s_c,
                    model_size=self.model_size,
                    native_size=native_crop_size,
                )
            except Exception as e:
                print(f"  WARNING: per-method uncertainty failed ({e}); "
                      f"falling back to legacy linear")
                unc_mm = 0.0
        elif rho_std > 0 and params.rho_eff > 0:
            unc_mm = defocus_uncertainty(
                result.sigma_native, params.rho_eff, params.sigma_0_eff,
                rho_std * params.scale_ratio, sigma_0_std * params.scale_ratio,
            )

        # Phase 7: get bounds_flag + method-aware defocus from CalibrationModel
        # when present in the checkpoint. SATURATED rows write nan to
        # defocus_mm; BELOW_FLOOR rows write 0.0. Legacy checkpoints
        # without a calibration_model leave bounds_flag = "IN_RANGE".
        bounds_flag = "IN_RANGE"
        inversion_method = "linear"
        cm = getattr(self, 'calibration_model', None)
        if cm is not None:
            try:
                z_cm, flag_cm = cm.inverse_at(
                    sigma_model_px=result.sigma_model,
                    s_inf_px_per_mm=s_c,
                    model_size=self.model_size,
                    native_size=native_crop_size,
                )
                bounds_flag = str(flag_cm.value)
                inversion_method = cm.method
                # Override defocus_mm with method-aware result. SATURATED
                # → nan (no honest answer); BELOW_FLOOR → 0.0.
                import math as _math
                if not _math.isnan(z_cm):
                    result.defocus_mm = z_cm
                else:
                    result.defocus_mm = float('nan')
            except Exception as e:
                print(f"  WARNING: CalibrationModel.inverse_at failed ({e}); "
                      f"using legacy linear result")

        return {
            "pred_norm": result.pred_norm,
            "sigma_model": result.sigma_model,
            "sigma_native": result.sigma_native,
            "defocus_mm": result.defocus_mm,
            "defocus_uncertainty_mm": unc_mm,
            "saturated": result.saturated,
            "clamped": result.clamped,
            "bounds_flag": bounds_flag,
            "inversion_method": inversion_method,
            "training_mode": self.training_mode,
            "model_path": str(self.settings.get("model_path", "")),
            "rho": rho,
            "sigma_0": sigma_0,
            "s_calib": s_calib,
            "s_c": s_c,
            "feather_px": feather_px,
            "crop_size": crop_size,
            "flatten_mode": str(self.settings.get("flatten_mode",
                                                    DEFAULT_FLATTEN_MODE)),
            "inner_margin_px": int(self.settings.get("inner_margin_px",
                                                       DEFAULT_INNER_MARGIN_PX)),
        }

    # ── Full pipeline convenience method ───────────────────────────────

    def process_cine(
        self, cine_path: Path, progress_cb=None,
    ) -> Dict[str, Any]:
        """Run the complete pipeline on a single .cine file.

        Args:
            cine_path: Path to .cine file.
            progress_cb: Optional callable(status_str, fraction_0_to_1).

        Returns:
            Dict with all intermediate and final results.
        """
        def _progress(msg, frac):
            if progress_cb is not None:
                progress_cb(msg, frac)

        _progress("Opening .cine file...", 0.0)
        cine_obj, frame_idx, geo, frame_gray = self.select_best_frame(cine_path)

        _progress(f"Selected frame {frame_idx}", 0.25)
        crop = self.extract_crop(frame_gray, geo)

        _progress("Boundary normalisation...", 0.50)
        norm_img, tensor_input = self.preprocess_crop(crop)

        _progress("Running model...", 0.75)
        native_size = max(crop.shape[0], crop.shape[1])
        results = self.run_inference(tensor_input, native_size)

        _progress("Done", 1.0)

        results["frame_idx"] = frame_idx
        results["frame_gray"] = frame_gray
        results["crop"] = crop
        results["norm_img"] = norm_img
        results["cine_name"] = cine_path.name
        results["geometry"] = geo

        return results

    # ── Batch processing ─────────────────────────────────────────────

    def process_folder(
        self, folder: Path, progress_cb=None,
    ) -> list:
        """Run the pipeline on every .cine file in a folder.

        Args:
            folder: Directory containing .cine files.
            progress_cb: Optional callable(status_str, file_index, total_files).

        Returns:
            List of result dicts. Failed files have an 'error' key.
        """
        cine_files = sorted(folder.glob("*.cine"), key=lambda p: p.name)
        if not cine_files:
            raise FileNotFoundError(f"No .cine files found in {folder}")

        all_results = []
        total = len(cine_files)

        for i, cine_path in enumerate(cine_files):
            if progress_cb is not None:
                progress_cb(f"Processing {i + 1}/{total}: {cine_path.name}", i, total)

            try:
                result = self.process_cine(cine_path)
                # Drop large arrays — not needed for batch CSV output
                for key in ("frame_gray", "crop", "norm_img", "geometry"):
                    result.pop(key, None)
                all_results.append(result)
            except Exception as e:
                all_results.append({
                    "cine_name": cine_path.name,
                    "error": str(e),
                })

        if progress_cb is not None:
            progress_cb("Batch complete", total, total)

        return all_results

    # ── Watch-folder helper ────────────────────────────────────────────

    @staticmethod
    def find_latest_cine(folder: Path) -> Optional[Path]:
        """Return the most recently modified .cine file in *folder*, or None."""
        cines = list(folder.glob("*.cine"))
        if not cines:
            return None
        return max(cines, key=lambda p: p.stat().st_mtime)
