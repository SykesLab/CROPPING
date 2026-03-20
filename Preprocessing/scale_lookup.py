"""
Scale lookup utility for mapping crop filenames to camera scale (px/mm).

Reads scales.csv which maps experiment number ranges + camera IDs to scale values.
Handles the CSV's quirks: missing dates, overlapping ranges, trailing commas, DUMMY entries.

Usage:
    scales = load_scales('scales.csv')
    scale = lookup_scale(scales, 'sphere1203g_crop.png')
    # Returns 50.2

    # Or from components:
    scale = lookup_scale_by_parts(scales, experiment_number=1203, camera_id='g')
"""

import csv
import re
import logging
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


# Type for a single scale entry: (start_exp, end_exp, camera_id, scale_px_per_mm)
ScaleEntry = Tuple[int, int, str, float]


def load_scales(csv_path) -> List[ScaleEntry]:
    """
    Load scales.csv into a list of (start_exp, end_exp, camera_id, scale) entries.

    Handles:
      - Missing date column (ignored anyway)
      - Trailing commas
      - DUMMY entries (filtered out)
      - Empty rows
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning(f"Scales CSV not found: {csv_path}")
        return []

    entries = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            # Strip whitespace and trailing empty fields
            row = [cell.strip() for cell in row if cell.strip()]

            if len(row) < 4:
                continue

            # Check for DUMMY entries
            if any('DUMMY' in cell.upper() for cell in row):
                continue

            # Columns: scaleDate, startExpNo, endExpNo, cameraID, scale
            # Date may be empty, so find the numeric columns
            try:
                # Try standard layout: date, start, end, cam, scale
                start = int(row[1])
                end = int(row[2])
                cam = row[3].strip().lower()
                scale = float(row[4]) if len(row) > 4 else float(row[3])
            except (ValueError, IndexError):
                try:
                    # Maybe date is missing, try: start, end, cam, scale
                    start = int(row[0])
                    end = int(row[1])
                    cam = row[2].strip().lower()
                    scale = float(row[3])
                except (ValueError, IndexError):
                    continue

            if scale > 0 and start <= end:
                entries.append((start, end, cam, scale))

    logger.info(f"Loaded {len(entries)} scale entries from {csv_path}")
    return entries


def extract_from_filename(filename: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Extract experiment number and camera ID from a crop filename.

    Handles formats:
      sphere0086v_crop.png  ->  (86, 'v')
      sphere1203g_crop.png  ->  (1203, 'g')
      sphere0080m.cine      ->  (80, 'm')
      7_mm_1.cine           ->  None (calibration file, not an experiment)

    Returns:
        (experiment_number, camera_id) or (None, None) if not parseable
    """
    stem = Path(filename).stem

    # Pattern: sphere{digits}{camera}...
    match = re.match(r'sphere(\d+)([a-z])', stem)
    if match:
        exp = int(match.group(1))
        cam = match.group(2)
        return exp, cam

    return None, None


def lookup_scale(entries: List[ScaleEntry], filename: str) -> Optional[float]:
    """
    Look up scale_px_per_mm for a crop filename.

    Args:
        entries: loaded scale entries from load_scales()
        filename: crop filename (e.g., 'sphere1203g_crop.png')

    Returns:
        scale_px_per_mm or None if not found
    """
    exp, cam = extract_from_filename(filename)
    if exp is None or cam is None:
        return None
    return lookup_scale_by_parts(entries, exp, cam)


def lookup_scale_by_parts(entries: List[ScaleEntry],
                          experiment_number: int,
                          camera_id: str) -> Optional[float]:
    """
    Look up scale_px_per_mm from experiment number and camera ID.

    If multiple ranges match (overlapping entries in CSV), returns the
    entry with the smallest range (most specific match).
    """
    camera_id = camera_id.lower()
    matches = []

    for start, end, cam, scale in entries:
        if cam == camera_id and start <= experiment_number <= end:
            range_size = end - start
            matches.append((range_size, scale))

    if not matches:
        return None

    # Return the most specific match (smallest range)
    matches.sort(key=lambda x: x[0])
    return matches[0][1]


def get_default_scale(camera_id: str) -> Optional[float]:
    """
    Fallback: return a default scale for known camera IDs.
    Used when scales.csv is unavailable or lookup fails.
    """
    defaults = {
        'g': 50.2,
        'v': 119.0,
        'm': 33.7,
        't': 199.2,
    }
    return defaults.get(camera_id.lower())


# Quick test when run directly
if __name__ == '__main__':
    import sys

    csv_path = Path(__file__).parent / 'scales.csv'
    entries = load_scales(csv_path)
    print(f"Loaded {len(entries)} entries from {csv_path}")

    test_files = [
        'sphere0086v_crop.png',
        'sphere0080m_crop.png',
        'sphere1203g_crop.png',
        'sphere0971t_crop.png',
        'sphere1750g_crop.png',
    ]

    for fn in test_files:
        exp, cam = extract_from_filename(fn)
        scale = lookup_scale(entries, fn)
        print(f"  {fn}: exp={exp}, cam={cam}, scale={scale}")
