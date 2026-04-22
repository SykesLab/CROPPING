"""
Shared training utilities.

Functions used by multiple training modules (train.py, test_model.py,
training_gui.py) to avoid duplication.
"""

import logging
from typing import Dict, List, Any

import numpy as np

logger = logging.getLogger(__name__)


def calculate_bin_weights_from_beta(config: Dict[str, Any]) -> List[float]:
    """Calculate bin weights from beta distribution parameters in config.

    Reads 'blur_distribution', 'beta_alpha', 'beta_beta' from the data
    section of the config. Returns 4 weights summing to 1.0 representing
    the training distribution across equal-width blur bins.

    Args:
        config: Training configuration dictionary with 'data' section.

    Returns:
        List of 4 floats summing to 1.0.
    """
    data_cfg = config.get('data', {})
    blur_distribution = data_cfg.get('blur_distribution', data_cfg.get('coc_distribution'))
    beta_alpha = data_cfg.get('beta_alpha')
    beta_beta = data_cfg.get('beta_beta')

    if blur_distribution is None:
        if beta_alpha is not None and beta_beta is not None:
            blur_distribution = 'weighted'
        else:
            logger.info("No distribution type or beta parameters found — using uniform weights")
            return [0.25, 0.25, 0.25, 0.25]

    if blur_distribution == 'uniform':
        logger.info("Using uniform distribution: equal bin weights")
        return [0.25, 0.25, 0.25, 0.25]

    if beta_alpha is None or beta_beta is None:
        logger.warning("Weighted distribution specified but beta parameters missing — falling back to uniform")
        return [0.25, 0.25, 0.25, 0.25]

    try:
        from scipy import stats

        beta_samples = stats.beta.rvs(beta_alpha, beta_beta, size=100000)

        bin_edges = [0.0, 0.25, 0.5, 0.75, 1.0]
        weights = []
        for i in range(4):
            count = np.sum((beta_samples >= bin_edges[i]) & (beta_samples < bin_edges[i + 1]))
            weights.append(count / len(beta_samples))

        total = sum(weights)
        weights = [w / total for w in weights]

        logger.info(
            f"Bin weights from beta({beta_alpha:.3f}, {beta_beta:.3f}): "
            + "-".join(f"{int(w * 100)}" for w in weights) + "%"
        )
        return weights

    except ImportError:
        logger.warning("scipy not available, using default weights [0.40, 0.30, 0.20, 0.10]")
        return [0.40, 0.30, 0.20, 0.10]
    except Exception as e:
        logger.warning(f"Error calculating bin weights: {e}")
        return [0.40, 0.30, 0.20, 0.10]
