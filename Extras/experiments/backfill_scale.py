"""
Backfill scale_px_per_mm and filename columns into sharp_crops.csv.

Reads scale_lookup.csv (folder, camera, exp_start, exp_end, scale_px_per_mm)
and annotates each row in sharp_crops.csv by matching on
  folder == row.folder  AND  camera == row.camera  AND  exp_start <= droplet_id <= exp_end
  AND NOT excluded

Adds two columns:
  filename       - basename of crop_path (used as key in synthetic_blur.py)
  scale_px_per_mm - camera px/mm for that crop
"""

from pathlib import Path
import pandas as pd

COURSEWORK = Path(__file__).parent.parent
SHARP_CSV = COURSEWORK / "Preprocessing/CROPPING/Preprocessing/OUTPUTNEW/Focus/sharp_crops.csv"
SCALE_CSV = Path(__file__).parent / "scale_lookup.csv"


def lookup_scale(row, scale_df):
    """Find scale for a single sharp_crops row. Returns float or None."""
    matches = scale_df[
        (scale_df['folder'] == row['folder']) &
        (scale_df['camera'] == row['camera']) &
        (scale_df['exp_start'] <= row['droplet_id']) &
        (scale_df['exp_end'] >= row['droplet_id']) &
        (scale_df['excluded'].isna() | (scale_df['excluded'] == ''))
    ]
    if matches.empty:
        return None
    return float(matches.iloc[-1]['scale_px_per_mm'])  # last-row-wins for overlaps


def main():
    crops = pd.read_csv(SHARP_CSV)
    scales = pd.read_csv(SCALE_CSV)

    print(f"Loaded {len(crops)} rows from sharp_crops.csv")
    print(f"Loaded {len(scales)} rows from scale_lookup.csv")

    # Add filename column
    crops['filename'] = crops['crop_path'].apply(lambda p: Path(p).name)

    # Add scale column
    crops['scale_px_per_mm'] = crops.apply(lambda r: lookup_scale(r, scales), axis=1)

    n_matched = crops['scale_px_per_mm'].notna().sum()
    n_missing = crops['scale_px_per_mm'].isna().sum()
    print(f"Matched: {n_matched}  |  Missing: {n_missing}")

    if n_missing > 0:
        missing = crops[crops['scale_px_per_mm'].isna()][['droplet_id', 'camera',
                                                      'folder']].drop_duplicates()
        print("Rows with no scale match:")
        print(missing.to_string(index=False))

    crops.to_csv(SHARP_CSV, index=False)
    print(f"Written: {SHARP_CSV}")


if __name__ == "__main__":
    main()
