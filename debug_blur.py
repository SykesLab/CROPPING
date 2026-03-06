"""
Debug script to visualize what's happening during erf blur measurement.

This helps diagnose why sharp images near focus are returning wrong σ values.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from blur_measurement import erf_edge, detect_sphere, get_sphere_mask


def debug_blur_measurement(image_path: str, save_dir: str = None):
    """
    Debug blur measurement for a single image.

    Shows:
    1. Detected sphere center and radius
    2. The radial rays being sampled
    3. Intensity profiles along each ray
    4. Erf fits and resulting sigma values
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load image: {image_path}")
        return

    img_float = img.astype(np.float32) / 255.0

    # Detect sphere
    center, radius = detect_sphere(img_float)
    if center is None:
        print("Could not detect sphere!")
        return

    cx, cy = center
    h, w = img.shape

    print(f"Image: {Path(image_path).name}")
    print(f"Image size: {w} x {h}")
    print(f"Detected center: ({cx}, {cy})")
    print(f"Detected radius: {radius} px")
    print()

    # Get the mask to visualize detection
    mask = get_sphere_mask(img_float)

    # Create visualization figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Original image with detected center/radius
    ax1 = axes[0, 0]
    ax1.imshow(img, cmap='gray')
    circle = plt.Circle((cx, cy), radius, fill=False, color='red', linewidth=2)
    ax1.add_patch(circle)
    ax1.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2)
    ax1.set_title(f'Detected: center=({cx},{cy}), r={radius}')
    ax1.axis('off')

    # 2. Thresholded mask
    ax2 = axes[0, 1]
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Otsu threshold mask')
    ax2.axis('off')

    # 3. Image with sample rays
    ax3 = axes[0, 2]
    ax3.imshow(img, cmap='gray')

    num_rays = 8  # Show fewer rays for clarity
    colors = plt.cm.rainbow(np.linspace(0, 1, num_rays))

    all_sigmas = []
    all_r_squared = []
    ray_data = []

    for i in range(num_rays):
        angle = 2 * np.pi * i / num_rays

        # Sample points along ray
        start_r = int(radius * 0.5)
        max_r = int(min(radius * 1.5, min(cx, cy, w - cx, h - cy)))
        r_values = np.arange(start_r, max_r)

        x_coords = (cx + r_values * np.cos(angle)).astype(int)
        y_coords = (cy + r_values * np.sin(angle)).astype(int)

        # Ensure within bounds
        valid = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
        r_valid = r_values[valid]
        x_valid = x_coords[valid]
        y_valid = y_coords[valid]

        if len(r_valid) < 10:
            continue

        # Draw ray on image
        ax3.plot(x_valid, y_valid, color=colors[i], linewidth=1, alpha=0.7)

        # Get intensity profile
        intensities = img_float[y_valid, x_valid]

        # Fit erf edge profile
        try:
            I_bg_init = np.median(intensities[-10:])
            I_sphere_init = np.median(intensities[:10])
            r_edge_init = radius
            sigma_init = 2.0

            popt, pcov = curve_fit(
                erf_edge, r_valid, intensities,
                p0=[I_bg_init, I_sphere_init, r_edge_init, sigma_init],
                bounds=(
                    [0, 0, radius * 0.5, 0.01],
                    [1, 1, radius * 1.5, 50]
                ),
                maxfev=1000
            )

            I_bg, I_sphere, r_edge, sigma_fit = popt

            # Calculate R-squared
            fitted = erf_edge(r_valid, *popt)
            ss_res = np.sum((intensities - fitted) ** 2)
            ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            ray_data.append({
                'angle': np.degrees(angle),
                'r_valid': r_valid,
                'intensities': intensities,
                'popt': popt,
                'r_squared': r_squared,
                'sigma': sigma_fit,
                'color': colors[i]
            })

            all_sigmas.append(sigma_fit)
            all_r_squared.append(r_squared)

        except (RuntimeError, ValueError) as e:
            ray_data.append({
                'angle': np.degrees(angle),
                'r_valid': r_valid,
                'intensities': intensities,
                'error': str(e),
                'color': colors[i]
            })

    ax3.set_title(f'Sample rays ({num_rays} rays)')
    ax3.axis('off')

    # 4. Intensity profiles
    ax4 = axes[1, 0]
    for data in ray_data:
        label = f"{data['angle']:.0f}°"
        if 'sigma' in data:
            label += f" σ={data['sigma']:.2f}"
        ax4.plot(data['r_valid'], data['intensities'], color=data['color'],
                 alpha=0.7, label=label)
    ax4.axvline(x=radius, color='black', linestyle='--', label=f'r={radius}')
    ax4.set_xlabel('Radial distance (pixels)')
    ax4.set_ylabel('Intensity')
    ax4.set_title('Raw intensity profiles')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Fitted erf profiles
    ax5 = axes[1, 1]
    for data in ray_data:
        if 'popt' in data:
            ax5.plot(data['r_valid'], data['intensities'], 'o', color=data['color'],
                     alpha=0.3, markersize=2)
            r_smooth = np.linspace(data['r_valid'][0], data['r_valid'][-1], 100)
            fitted = erf_edge(r_smooth, *data['popt'])
            ax5.plot(r_smooth, fitted, color=data['color'], linewidth=2,
                     label=f"σ={data['sigma']:.2f}, R²={data['r_squared']:.2f}")
    ax5.axvline(x=radius, color='black', linestyle='--')
    ax5.set_xlabel('Radial distance (pixels)')
    ax5.set_ylabel('Intensity')
    ax5.set_title('Erf fits')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')

    if all_sigmas:
        summary = f"""
Detected sphere:
  Center: ({cx}, {cy})
  Radius: {radius} px

Fit results ({len(all_sigmas)} successful fits):
  σ values: {[f'{s:.2f}' for s in all_sigmas]}
  σ median: {np.median(all_sigmas):.3f} px
  σ mean: {np.mean(all_sigmas):.3f} px
  σ std: {np.std(all_sigmas):.3f} px
  σ min: {np.min(all_sigmas):.3f} px
  σ max: {np.max(all_sigmas):.3f} px

R² values: {[f'{r:.2f}' for r in all_r_squared]}
  Mean R²: {np.mean(all_r_squared):.3f}

Image statistics:
  Min: {img.min()}
  Max: {img.max()}
  Mean: {img.mean():.1f}
"""
    else:
        summary = "No successful erf fits!"

    ax6.text(0.1, 0.9, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / f"debug_{Path(image_path).stem}.png"
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()

    # Print detailed fit info
    print("\nDetailed fit results:")
    print("-" * 60)
    for data in ray_data:
        if 'popt' in data:
            I_bg, I_sphere, r_edge, sigma = data['popt']
            print(f"Ray {data['angle']:5.0f}°: σ={sigma:6.3f} px, R²={data['r_squared']:.3f}, "
                  f"I_bg={I_bg:.3f}, I_sphere={I_sphere:.3f}, r_edge={r_edge:.1f}")
        else:
            print(f"Ray {data['angle']:5.0f}°: FIT FAILED - {data.get('error', 'unknown')}")

    return ray_data


def debug_multiple_images(image_dir: str, pattern: str = "*.png"):
    """Debug blur measurement for multiple images in a directory."""
    image_dir = Path(image_dir)
    images = sorted(image_dir.glob(pattern))

    print(f"Found {len(images)} images")

    for img_path in images:
        print("\n" + "=" * 60)
        debug_blur_measurement(str(img_path))
        input("Press Enter for next image...")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        debug_blur_measurement(image_path)
    else:
        print("Usage: python debug_blur.py <image_path>")
        print("Or import and call debug_blur_measurement(path) from Python")
