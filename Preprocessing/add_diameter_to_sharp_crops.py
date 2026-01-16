"""Add diameter_px column to existing sharp_crops.csv retrospectively.

Reads the sharp_crops.csv and merges geometry data from the summary CSVs
to compute diameter_px = y_bottom - y_top for each crop.

Functions:
    load_sharp_crops: Load sharp crops CSV with validation.
    find_summary_csvs: Locate all summary CSV files.
    load_and_combine_summaries: Load and combine all summary CSVs.
    merge_diameter_data: Merge geometry data to add diameter_px.
    reorder_columns: Reorder columns to place diameter_px after y_sphere.
    save_with_backup: Save updated CSV with backup of original.
    main: Entry point for the script.

Usage:
    python add_diameter_to_sharp_crops.py
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_sharp_crops(csv_path: Path) -> pd.DataFrame:
    """Load sharp crops CSV with validation.

    Args:
        csv_path: Path to sharp_crops.csv file.

    Returns:
        DataFrame with sharp crops data.

    Raises:
        FileNotFoundError: If CSV file doesn't exist.
        pd.errors.EmptyDataError: If CSV is empty.
        ValueError: If CSV cannot be parsed.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Sharp crops CSV not found: {csv_path}")

    logger.info(f"Reading {csv_path}...")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Sharp crops CSV is empty: {csv_path}")
    except Exception as e:
        raise ValueError(f"Failed to parse CSV {csv_path}: {e}")

    if df.empty:
        raise pd.errors.EmptyDataError(f"Sharp crops CSV contains no data: {csv_path}")

    logger.info(f"Found {len(df)} sharp crops")
    return df


def find_summary_csvs(output_root: Path) -> List[Path]:
    """Find all summary CSV files in output directory.

    Args:
        output_root: Root output directory to search.

    Returns:
        List of paths to summary CSV files.
    """
    if not output_root.exists():
        logger.warning(f"Output directory does not exist: {output_root}")
        return []

    summary_csvs = list(output_root.rglob("*_summary.csv"))
    logger.info(f"Found {len(summary_csvs)} summary CSV files")
    return summary_csvs


def load_and_combine_summaries(summary_paths: List[Path]) -> pd.DataFrame:
    """Load and combine all summary CSVs.

    Args:
        summary_paths: List of paths to summary CSV files.

    Returns:
        Combined DataFrame from all summaries.

    Raises:
        ValueError: If no valid summary CSVs could be loaded.
    """
    if not summary_paths:
        raise ValueError("No summary CSV files provided")

    all_summaries: List[pd.DataFrame] = []
    failed_count = 0

    for csv_path in summary_paths:
        try:
            df = pd.read_csv(csv_path)
            all_summaries.append(df)
        except Exception as e:
            logger.warning(f"Could not read {csv_path}: {e}")
            failed_count += 1

    if not all_summaries:
        raise ValueError(
            f"No valid summary CSVs could be loaded "
            f"(attempted {len(summary_paths)}, failed {failed_count})"
        )

    if failed_count > 0:
        logger.warning(f"Failed to load {failed_count} summary CSV files")

    combined = pd.concat(all_summaries, ignore_index=True)
    logger.info(f"Combined {len(combined)} total crops from summaries")
    return combined


def merge_diameter_data(
    sharp_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge geometry data to add diameter_px column.

    Args:
        sharp_df: Sharp crops DataFrame.
        summary_df: Combined summary DataFrame with geometry.

    Returns:
        Merged DataFrame with diameter_px column.

    Raises:
        ValueError: If required columns are missing.
    """
    # Validate required columns
    required_sharp_cols = ['crop_path']
    required_summary_cols = ['crop_path', 'y_top', 'y_bottom']

    missing_sharp = [c for c in required_sharp_cols if c not in sharp_df.columns]
    if missing_sharp:
        raise ValueError(f"Sharp crops CSV missing columns: {missing_sharp}")

    missing_summary = [c for c in required_summary_cols if c not in summary_df.columns]
    if missing_summary:
        raise ValueError(f"Summary CSVs missing columns: {missing_summary}")

    logger.info("Merging geometry data...")

    # Select columns to merge (include y_sphere if available)
    merge_cols = ['crop_path', 'y_top', 'y_bottom']
    if 'y_sphere' in summary_df.columns:
        merge_cols.append('y_sphere')

    merged_df = sharp_df.merge(
        summary_df[merge_cols],
        on='crop_path',
        how='left'
    )

    # Compute diameter_px
    merged_df['diameter_px'] = merged_df['y_bottom'] - merged_df['y_top']

    # Report any missing matches
    missing_count = merged_df['diameter_px'].isna().sum()
    if missing_count > 0:
        logger.warning(
            f"{missing_count} crops ({missing_count/len(merged_df)*100:.1f}%) "
            f"could not be matched with geometry data"
        )

    return merged_df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to place diameter_px after y_sphere.

    Args:
        df: DataFrame with diameter_px column.

    Returns:
        DataFrame with reordered columns.
    """
    if 'diameter_px' not in df.columns:
        return df

    cols = list(df.columns)
    cols.remove('diameter_px')

    # Insert after y_sphere if present, otherwise append
    if 'y_sphere' in cols:
        sphere_idx = cols.index('y_sphere')
        cols.insert(sphere_idx + 1, 'diameter_px')
    else:
        cols.append('diameter_px')

    return df[cols]


def save_with_backup(
    df: pd.DataFrame,
    original_path: Path,
    backup_suffix: str = "_backup",
) -> None:
    """Save updated CSV with backup of original.

    Args:
        df: DataFrame to save.
        original_path: Path to original file (will be backed up and replaced).
        backup_suffix: Suffix for backup file name.

    Raises:
        IOError: If file operations fail.
    """
    parent_dir = original_path.parent
    stem = original_path.stem
    suffix = original_path.suffix

    # Create paths
    temp_path = parent_dir / f"{stem}_with_diameter{suffix}"
    backup_path = parent_dir / f"{stem}{backup_suffix}{suffix}"

    try:
        # Save to temp file first
        df.to_csv(temp_path, index=False)
        logger.info(f"Saved updated CSV to: {temp_path}")

        # Create backup of original
        if original_path.exists():
            if backup_path.exists():
                backup_path.unlink()  # Remove old backup
            original_path.rename(backup_path)
            logger.info(f"Original backed up to: {backup_path}")

        # Move temp to original location
        temp_path.rename(original_path)
        logger.info(f"Updated {original_path.name} with diameter_px column")

    except IOError as e:
        logger.error(f"File operation failed: {e}")
        # Attempt cleanup
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise


def print_statistics(df: pd.DataFrame) -> None:
    """Print diameter statistics.

    Args:
        df: DataFrame with diameter_px column.
    """
    if 'diameter_px' not in df.columns:
        logger.warning("No diameter_px column found for statistics")
        return

    valid_diameters = df['diameter_px'].dropna()

    logger.info(f"\nTotal crops: {len(df)}")
    logger.info(f"Crops with diameter: {len(valid_diameters)}")
    logger.info(f"\nDiameter statistics (pixels):")
    logger.info(f"  Min:    {valid_diameters.min():.1f}")
    logger.info(f"  Max:    {valid_diameters.max():.1f}")
    logger.info(f"  Mean:   {valid_diameters.mean():.1f}")
    logger.info(f"  Median: {valid_diameters.median():.1f}")
    logger.info(f"  Std:    {valid_diameters.std():.1f}")


def main(
    preprocessing_root: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    """Main entry point for diameter addition script.

    Args:
        preprocessing_root: Root directory of preprocessing pipeline.
                          If None, uses script's parent directory.
        dry_run: If True, don't modify files (just show what would happen).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    if preprocessing_root is None:
        preprocessing_root = Path(__file__).parent

    output_root = preprocessing_root / "OUTPUT"
    focus_dir = output_root / "Focus"
    sharp_crops_path = focus_dir / "sharp_crops.csv"

    try:
        # Load sharp crops
        sharp_df = load_sharp_crops(sharp_crops_path)

        # Find and load summary CSVs
        summary_csvs = find_summary_csvs(output_root)
        if not summary_csvs:
            logger.error("No summary CSVs found - cannot add diameter data")
            return 1

        combined_summary = load_and_combine_summaries(summary_csvs)

        # Merge diameter data
        merged_df = merge_diameter_data(sharp_df, combined_summary)

        # Reorder columns
        merged_df = reorder_columns(merged_df)

        # Print statistics
        print_statistics(merged_df)

        # Save with backup
        if dry_run:
            logger.info("\n[DRY RUN] Would save to: {sharp_crops_path}")
        else:
            save_with_backup(merged_df, sharp_crops_path)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Data error: {e}")
        return 1
    except IOError as e:
        logger.error(f"I/O error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
