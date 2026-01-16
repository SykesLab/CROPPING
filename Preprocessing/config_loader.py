"""YAML configuration loader for the preprocessing pipeline.

Loads configuration from preprocessing_config.yaml and provides
typed access to all settings with validation and sensible defaults.

Usage:
    from config_loader import load_config

    config = load_config()
    print(config.crop.max_cnn_size)  # 512
    print(config.paths.output_root)  # Path('./OUTPUT')
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PathsConfig:
    """Path configuration settings."""
    cine_root: Path = field(default_factory=lambda: Path("./data"))
    output_root: Path = field(default_factory=lambda: Path("./OUTPUT"))


@dataclass
class CropConfig:
    """Crop size configuration."""
    max_cnn_size: int = 512
    min_cnn_size: int = 64
    safety_pixels: int = 3


@dataclass
class SamplingConfig:
    """Sampling configuration."""
    cine_step: int = 10


@dataclass
class GeometryConfig:
    """Geometry analysis configuration."""
    min_area: int = 50
    sphere_width_ratio: float = 0.30
    sphere_center_tolerance: float = 0.35


@dataclass
class BestFrameConfig:
    """Best frame selection configuration."""
    n_candidates: int = 20
    darkness_threshold_percentile: float = 70.0
    darkness_weight: float = 0.05


@dataclass
class CalibrationConfig:
    """Calibration configuration."""
    percentile: float = 5.0


@dataclass
class FocusConfig:
    """Focus metrics configuration."""
    enabled: bool = True
    primary_metric: str = "laplacian_var"
    sharp_threshold: Optional[float] = None
    blur_threshold: Optional[float] = None


@dataclass
class PipelineConfig:
    """Complete pipeline configuration.

    Contains all configuration sections as nested dataclasses,
    providing typed access to all pipeline parameters.

    Attributes:
        paths: Path configuration for input/output directories.
        crop: Crop size bounds and safety margins.
        sampling: Sampling rate for processing.
        geometry: Parameters for droplet/sphere detection.
        best_frame: Parameters for optimal frame selection.
        calibration: Parameters for crop size calibration.
        focus: Parameters for focus quality analysis.
    """
    paths: PathsConfig = field(default_factory=PathsConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    best_frame: BestFrameConfig = field(default_factory=BestFrameConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    focus: FocusConfig = field(default_factory=FocusConfig)


def load_config(config_path: Optional[Path] = None) -> PipelineConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default location
                    (preprocessing_config.yaml in the same directory).

    Returns:
        PipelineConfig with all settings loaded from YAML,
        falling back to defaults for missing values.

    Example:
        >>> config = load_config()
        >>> config.crop.max_cnn_size
        512
        >>> config.focus.primary_metric
        'laplacian_var'
    """
    if config_path is None:
        config_path = Path(__file__).parent / "preprocessing_config.yaml"

    if not config_path.exists():
        logger.warning(
            f"Config file not found: {config_path}. Using default values."
        )
        return PipelineConfig()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML config: {e}")
        raise
    except IOError as e:
        logger.error(f"Failed to read config file: {e}")
        raise

    if raw_config is None:
        logger.warning("Config file is empty. Using default values.")
        return PipelineConfig()

    return _parse_config(raw_config)


def _parse_config(raw: Dict[str, Any]) -> PipelineConfig:
    """Parse raw YAML dict into typed config objects.

    Args:
        raw: Dictionary loaded from YAML file.

    Returns:
        PipelineConfig with parsed values.
    """
    config = PipelineConfig()

    # Parse paths section
    if 'paths' in raw:
        paths_raw = raw['paths']
        config.paths = PathsConfig(
            cine_root=Path(paths_raw.get('cine_root', './data')),
            output_root=Path(paths_raw.get('output_root', './OUTPUT')),
        )

    # Parse crop section
    if 'crop' in raw:
        crop_raw = raw['crop']
        config.crop = CropConfig(
            max_cnn_size=int(crop_raw.get('max_cnn_size', 512)),
            min_cnn_size=int(crop_raw.get('min_cnn_size', 64)),
            safety_pixels=int(crop_raw.get('safety_pixels', 3)),
        )

    # Parse sampling section
    if 'sampling' in raw:
        sampling_raw = raw['sampling']
        config.sampling = SamplingConfig(
            cine_step=int(sampling_raw.get('cine_step', 10)),
        )

    # Parse geometry section
    if 'geometry' in raw:
        geom_raw = raw['geometry']
        config.geometry = GeometryConfig(
            min_area=int(geom_raw.get('min_area', 50)),
            sphere_width_ratio=float(geom_raw.get('sphere_width_ratio', 0.30)),
            sphere_center_tolerance=float(geom_raw.get('sphere_center_tolerance', 0.35)),
        )

    # Parse best_frame section
    if 'best_frame' in raw:
        bf_raw = raw['best_frame']
        config.best_frame = BestFrameConfig(
            n_candidates=int(bf_raw.get('n_candidates', 20)),
            darkness_threshold_percentile=float(bf_raw.get('darkness_threshold_percentile', 70.0)),
            darkness_weight=float(bf_raw.get('darkness_weight', 0.05)),
        )

    # Parse calibration section
    if 'calibration' in raw:
        cal_raw = raw['calibration']
        config.calibration = CalibrationConfig(
            percentile=float(cal_raw.get('percentile', 5.0)),
        )

    # Parse focus section
    if 'focus' in raw:
        focus_raw = raw['focus']
        sharp_thresh = focus_raw.get('sharp_threshold')
        blur_thresh = focus_raw.get('blur_threshold')

        config.focus = FocusConfig(
            enabled=bool(focus_raw.get('enabled', True)),
            primary_metric=str(focus_raw.get('primary_metric', 'laplacian_var')),
            sharp_threshold=float(sharp_thresh) if sharp_thresh is not None else None,
            blur_threshold=float(blur_thresh) if blur_thresh is not None else None,
        )

    return config


def save_config(config: PipelineConfig, config_path: Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: PipelineConfig to save.
        config_path: Path to save the config file.

    Raises:
        IOError: If file cannot be written.
    """
    config_dict = {
        'paths': {
            'cine_root': str(config.paths.cine_root),
            'output_root': str(config.paths.output_root),
        },
        'crop': {
            'max_cnn_size': config.crop.max_cnn_size,
            'min_cnn_size': config.crop.min_cnn_size,
            'safety_pixels': config.crop.safety_pixels,
        },
        'sampling': {
            'cine_step': config.sampling.cine_step,
        },
        'geometry': {
            'min_area': config.geometry.min_area,
            'sphere_width_ratio': config.geometry.sphere_width_ratio,
            'sphere_center_tolerance': config.geometry.sphere_center_tolerance,
        },
        'best_frame': {
            'n_candidates': config.best_frame.n_candidates,
            'darkness_threshold_percentile': config.best_frame.darkness_threshold_percentile,
            'darkness_weight': config.best_frame.darkness_weight,
        },
        'calibration': {
            'percentile': config.calibration.percentile,
        },
        'focus': {
            'enabled': config.focus.enabled,
            'primary_metric': config.focus.primary_metric,
            'sharp_threshold': config.focus.sharp_threshold,
            'blur_threshold': config.focus.blur_threshold,
        },
    }

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to: {config_path}")
    except IOError as e:
        logger.error(f"Failed to save config: {e}")
        raise


# Module-level singleton for convenience
_cached_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """Get the cached configuration, loading if necessary.

    Returns:
        Cached PipelineConfig instance.
    """
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config()
    return _cached_config


def reload_config(config_path: Optional[Path] = None) -> PipelineConfig:
    """Reload configuration from file, updating the cache.

    Args:
        config_path: Optional path to config file.

    Returns:
        Newly loaded PipelineConfig instance.
    """
    global _cached_config
    _cached_config = load_config(config_path)
    return _cached_config
