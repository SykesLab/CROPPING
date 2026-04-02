"""
Test script to verify blur measurement fixes.

Run this on a few images to see if the erf fitting now works properly.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from blur_measurement import measure_blur_erf, detect_sphere


def test_image(image_path: str):
    """Test blur measurement on a single image with verbose output."""
    print(f"\n{'='*60}")
    print(f"Testing: {Path(image_path).name}")
    print('='*60)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Could not load image")
        return

    print(f"Image size: {img.shape[1]} x {img.shape[0]}")
    print(f"Intensity range: {img.min()} - {img.max()}")

    # Run measurement with verbose output
    result = measure_blur_erf(img, verbose=True)

    print(f"\n--- RESULT ---")
    print(f"σ = {result.sigma:.4f} px")
    print(f"Confidence = {result.confidence:.4f}")
    print(f"Method = {result.method}")

    if 'error' in result.details:
        print(f"Error: {result.details['error']}")
    else:
        print(f"Rays used: {result.details['num_rays_used']}")
        print(f"σ std: {result.details['sigma_std']:.4f}")
        print(f"Mean R²: {result.details['mean_r_squared']:.4f}")
        if result.details.get('all_sigmas'):
            all_s = result.details['all_sigmas']
            print(f"All σ values: {[f'{s:.2f}' for s in sorted(all_s)]}")


def test_directory(image_dir: str, pattern: str = "*.png"):
    """Test all images in a directory."""
    image_dir = Path(image_dir)
    images = sorted(image_dir.glob(pattern))

    print(f"Found {len(images)} images matching '{pattern}'")

    results = []
    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        result = measure_blur_erf(img, verbose=False)
        results.append((img_path.name, result.sigma, result.confidence))

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"{'Image':<30} {'σ (px)':>10} {'Confidence':>12}")
    print('-'*60)
    for name, sigma, conf in results:
        print(f"{name:<30} {sigma:>10.3f} {conf:>12.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if Path(path).is_dir():
            pattern = sys.argv[2] if len(sys.argv) > 2 else "*.png"
            test_directory(path, pattern)
        else:
            test_image(path)
    else:
        print("Usage:")
        print("  python test_blur_fix.py <image_path>       # Test single image")
        print("  python test_blur_fix.py <directory> [pattern]  # Test all images")
        print()
        print("Example:")
        print("  python test_blur_fix.py calibration_images/")
        print("  python test_blur_fix.py calibration_images/ *.tif")
