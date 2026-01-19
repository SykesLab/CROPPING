"""
YAML configuration loader with typed dataclasses.

Loads settings from preprocessing_config.yaml with sensible defaults
for any missing values.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PathsConfig:
    cine_root: Path = field(default_factory=lambda: Path("./data"))
    output_root: Path = field(default_factory=lambda: Path("./OUTPUT"))


@dataclass
class CropConfig:
    max_cnn_size: int = 512
    min_cnn_size: int = 64
    safety_pixels: int = 3


@dataclass
class SamplingConfig:
    cine_step: int = 10


@dataclass
class GeometryConfig:
    min_area: int = 50
    sphere_width_ratio: float = 0.30
    sphere_center_tolerance: float = 0.35


@dataclass
class BestFrameConfig:
    n_candidates: int = 20
    darkness_threshold_percentile: float = 70.0
    darkness_weight: float = 0.05


@dataclass
class CalibrationConfig:
    percentile: float = 5.0


@dataclass
class FocusConfig:
    enabled: bool = True
    primary_metric: str = "laplacian_var"
    sharp_threshold: Optional[float] = None
    blur_threshold: Optional[float] = None


@dataclass
class PipelineConfig:
    """All pipeline settings grouped by category."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    best_frame: BestFrameConfig = field(default_factory=BestFrameConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    focus: FocusConfig = field(default_factory=FocusConfig)


def load_config(config_path: Optional[Path] = None) -> PipelineConfig:
    """Load configuration from YAML, falling back to defaults for missing values."""
    if config_path is None:
        config_path = Path(__file__).parent / "preprocessing_config.yaml"

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
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
        logger.warning("Config file is empty. Using defaults.")
        return PipelineConfig()

    return _parse_config(raw_config)


def _parse_config(raw: Dict[str, Any]) -> PipelineConfig:
    """Parse raw YAML dict into typed config objects."""
    config = PipelineConfig()

    if 'paths' in raw:
        p = raw['paths']
        config.paths = PathsConfig(
            cine_root=Path(p.get('cine_root', './data')),
            output_root=Path(p.get('output_root', './OUTPUT')),
        )

    if 'crop' in raw:
        c = raw['crop']
        config.crop = CropConfig(
            max_cnn_size=int(c.get('max_cnn_size', 512)),
            min_cnn_size=int(c.get('min_cnn_size', 64)),
            safety_pixels=int(c.get('safety_pixels', 3)),
        )

    if 'sampling' in raw:
        s = raw['sampling']
        config.sampling = SamplingConfig(cine_step=int(s.get('cine_step', 10)))

    if 'geometry' in raw:
        g = raw['geometry']
        config.geometry = GeometryConfig(
            min_area=int(g.get('min_area', 50)),
            sphere_width_ratio=float(g.get('sphere_width_ratio', 0.30)),
            sphere_center_tolerance=float(g.get('sphere_center_tolerance', 0.35)),
        )

    if 'best_frame' in raw:
        bf = raw['best_frame']
        config.best_frame = BestFrameConfig(
            n_candidates=int(bf.get('n_candidates', 20)),
            darkness_threshold_percentile=float(bf.get('darkness_threshold_percentile', 70.0)),
            darkness_weight=float(bf.get('darkness_weight', 0.05)),
        )

    if 'calibration' in raw:
        cal = raw['calibration']
        config.calibration = CalibrationConfig(percentile=float(cal.get('percentile', 5.0)))

    if 'focus' in raw:
        f = raw['focus']
        sharp = f.get('sharp_threshold')
        blur = f.get('blur_threshold')
        config.focus = FocusConfig(
            enabled=bool(f.get('enabled', True)),
            primary_metric=str(f.get('primary_metric', 'laplacian_var')),
            sharp_threshold=float(sharp) if sharp is not None else None,
            blur_threshold=float(blur) if blur is not None else None,
        )

    return config


def save_config(config: PipelineConfig, config_path: Path) -> None:
    """Save configuration to YAML file."""
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
        'sampling': {'cine_step': config.sampling.cine_step},
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
        'calibration': {'percentile': config.calibration.percentile},
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


_cached_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """Get the cached configuration, loading if necessary."""
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config()
    return _cached_config


def reload_config(config_path: Optional[Path] = None) -> PipelineConfig:
    """Reload configuration from file, updating the cache."""
    global _cached_config
    _cached_config = load_config(config_path)
    return _cached_config
