"""
Defocus Estimation Training GUI

A unified GUI for the training workflow:
1. Scan sharp crops directory for image statistics
2. Configure optical parameters (global or per-folder)
3. Generate synthetic training data
4. Train the defocus estimation model

Usage:
    python training_gui.py
     
Dependencies:
    pip install pyyaml numpy opencv-python torch torchvision pandas tqdm
"""

import sys

# =============================================================================
# Dependency Check
# =============================================================================


def check_dependencies():
    """Check for required dependencies and provide install instructions."""
    missing = []

    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    try:
        import tqdm
    except ImportError:
        missing.append("tqdm")

    if missing:
        print("=" * 60)
        print("ERROR: Missing required dependencies:")
        for pkg in missing:
            print(f"  - {pkg}")
        print()
        print("Install with:")
        print(f"  pip install {' '.join(missing)}")
        print()
        print("Or install all dependencies:")
        print("  pip install pyyaml numpy opencv-python pandas tqdm torch torchvision")
        print("=" * 60)
        sys.exit(1)


check_dependencies()

# =============================================================================
# Imports (after dependency check)
# =============================================================================
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import webbrowser
import yaml
import numpy as np
import cv2

# Add parent directory to path for imports (fallback for non-package usage)
_parent = str(Path(__file__).parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)


@dataclass
class OpticalConfig:
    """Optical configuration for a folder/setup."""
    folder_name: str = ""
    # Lens parameters (user enters these)
    focal_length_mm: float = 200.0
    f_number: float = 4.0
    focus_distance_mm: float = 400.0
    # Calculated from above
    aperture_diameter_mm: float = 50.0
    imaging_distance_mm: float = 400.0
    # From camera metadata
    pixel_size_mm: float = 0.02
    sensor_width_px: int = 1024
    sensor_height_px: int = 1024
    # Blur synthesis
    rho: float = 1.0
    # Training range
    defocus_range_min_mm: float = -12.0
    defocus_range_max_mm: float = 12.0

    def update_calculated(self):
        """Update calculated parameters from user inputs."""
        self.aperture_diameter_mm = self.focal_length_mm / self.f_number
        # Lens equation: 1/F = 1/d0 + 1/u0 -> u0 = 1/(1/F - 1/d0)
        try:
            inv_u0 = 1.0 / self.focal_length_mm - 1.0 / self.focus_distance_mm
            if inv_u0 > 0:
                self.imaging_distance_mm = 1.0 / inv_u0
            else:
                self.imaging_distance_mm = self.focus_distance_mm
        except ZeroDivisionError:
            self.imaging_distance_mm = self.focus_distance_mm

    def to_yaml_dict(self) -> dict:
        """Convert to YAML-compatible dict for training_config.yaml."""
        self.update_calculated()
        # Use actual image size from scanned folders
        image_size = min(self.sensor_width_px, self.sensor_height_px)
        return {
            'optics': {
                'focal_length_mm': self.focal_length_mm,
                'focus_distance_mm': self.focus_distance_mm,
                'imaging_distance_mm': self.imaging_distance_mm,
                'aperture_diameter_mm': self.aperture_diameter_mm,
                'pixel_size_mm': self.pixel_size_mm,
                'sensor_width_px': self.sensor_width_px,
                'sensor_height_px': self.sensor_height_px,
            },
            'blur': {
                'rho': self.rho,
                'kernel_radius_factor': 4.0,
            },
            'data': {
                'defocus_range_mm': [self.defocus_range_min_mm, self.defocus_range_max_mm],
                'droplet_diameter_range_px': [10, 150],
                'num_samples': 50000,
                'image_size_px': image_size,
            },
            'training': {
                'batize': 50,
                'epochs_dme': 400,
                'learning_rate': 0.0002,
            }
        }


# =============================================================================
# Sharp Crops Scanner (simple directory scan + optional CSV stats)
# =============================================================================
@dataclass
class FolderStats:
    """Statistics for a folder of sharp crop images."""
    folder_name: str = ""
    folder_path: Path = None
    num_images: int = 0
    image_width: int = 0
    image_height: int = 0
    size_consistent: bool = True
    sizes_found: Dict[Tuple[int, int], int] = field(default_factory=dict)  # {(w,h): count}
    # Focus metrics from CSV (if available)
    has_focus_metrics: bool = False
    mean_laplacian: float = 0.0
    min_laplacian: float = 0.0
    max_laplacian: float = 0.0
    mean_tenengrad: float = 0.0
    # Scale / blur / diameter / camera from sharp_crops.csv
    has_csv_metadata: bool = False
    camera: str = ""
    mean_scale_px_per_mm: float = 0.0
    min_scale_px_per_mm: float = 0.0
    max_scale_px_per_mm: float = 0.0
    mean_native_blur: float = 0.0
    min_native_blur: float = 0.0
    max_native_blur: float = 0.0
    mean_diameter_px: float = 0.0
    min_diameter_px: float = 0.0
    max_diameter_px: float = 0.0


class SharpCropsScanner:
    """Scans sharp crops directories."""

    def __init__(self):
        self.supported_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']

    def scan_root(self, root: Path, log_callback=None, camera_filter: str="all") ->Dict[str,
            FolderStats]:
        """Scan directory for image folders."""
        root = Path(root)
        results = {}

        if log_callback:
            log_callback(f"Scanning: {root}")

        # Look for sharp_crops.csv in root or parent
        csv_path = self._find_sharp_crops_csv(root)
        csv_data = None
        if csv_path:
            if log_callback:
                log_callback(f"Found: {csv_path.name}")
            csv_data = self._load_csv_data(csv_path)

        # Check if root itself has images (no subfolders case)
        root_images = self._get_image_files(root)

        # Check for subfolders - supports nested structure: root/material/camera/images
        material_folders = sorted([d for d in root.iterdir() if d.is_dir()])
        folders_with_images = []

        for material_folder in material_folders:
            # Check if this folder has camera subfolders (g, m, v)
            camera_folders = sorted([d for d in material_folder.iterdir() if d.is_dir()])

            if camera_folders:
                # Nested structure: material/camera/images
                for camera_folder in camera_folders:
                    # Filter by camera type if specified
                    if camera_filter and camera_filter != "all":
                        if camera_folder.name.lower() != camera_filter.lower():
                            continue

                    images = self._get_image_files(camera_folder)
                    # Also check for crops/ subfolder (material/camera/crops/)
                    crops_subfolder = camera_folder / "crops"
                    if not images and crops_subfolder.is_dir():
                        images = self._get_image_files(crops_subfolder)
                    if images:
                        # Use combined name: material/camera
                        folder_name = f"{material_folder.name}/{camera_folder.name}"
                        folders_with_images.append((camera_folder, images, folder_name,
                                                    material_folder.name, camera_folder.name))
            else:
                # Check if material folder itself has images (flat structure)
                images = self._get_image_files(material_folder)
                if images:
                    # Filter by folder name if it matches camera filter
                    if camera_filter and camera_filter != "all":
                        if material_folder.name.lower() != camera_filter.lower():
                            continue
                    folders_with_images.append((material_folder, images, material_folder.name,
                                                material_folder.name, None))

        if camera_filter and camera_filter != "all" and log_callback:
            log_callback(
                f"Filtering to camera '{camera_filter}': {len(folders_with_images)} matching folders")

        if folders_with_images:
            # Has folders with images - scan each
            for folder_path, image_files, folder_name, csv_key, cam in folders_with_images:
                stats = self._analyze_folder(folder_path, image_files, csv_data, log_callback,
                                             csv_key=csv_key, camera_filter=cam)
                results[folder_name] = stats

        elif root_images:
            # Root itself has images (single folder case)
            stats = self._analyze_folder(root, root_images, csv_data, log_callback,
                                         csv_key=root.name)
            results[root.name] = stats

        else:
            if log_callback:
                log_callback("No images found!")

        return results

    def _find_sharp_crops_csv(self, root: Path) -> Optional[Path]:
        """Look for sharp_crops.csv in root or parent directories (up to 3 levels)."""
        search_dir = root

        # Search up to 3 levels up
        for _ in range(4):  # root + 3 parents
            # Check exact name
            csv_path = search_dir / "sharp_crops.csv"
            if csv_path.exists():
                return csv_path

            # Check for timestamped version (find most recent)
            csv_candidates = list(search_dir.glob("*sharp_crops.csv"))
            if csv_candidates:
                # Return most recently modified CSV
                return max(csv_candidates, key=lambda p: p.stat().st_mtime)

            # Go up one level
            if search_dir.parent and search_dir.parent != search_dir:
                search_dir = search_dir.parent
            else:
                break

        return None

    def _load_csv_data(self, csv_path: Path) -> Optional[Dict[str, List[Dict]]]:
        """Load CSV and group by folder."""
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)

            # Group by folder
            data_by_folder = {}
            for folder_name in df['folder'].unique():
                folder_df = df[df['folder'] == folder_name]
                data_by_folder[folder_name] = folder_df.to_dict('records')

            return data_by_folder
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None

    def _get_image_files(self, folder: Path) -> List[Path]:
        """Get all image files in a folder (non-recursive)."""
        images = []
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() in self.supported_extensions:
                images.append(f)
        return sorted(images)

    def _analyze_folder(self, folder: Path, image_files: List[Path],
                        csv_data: Optional[Dict], log_callback=None,
                        csv_key: Optional[str] = None,
                        camera_filter: Optional[str] = None) -> FolderStats:
        """Analyze all images in a folder for size consistency."""
        stats = FolderStats(
            folder_name=folder.name,
            folder_path=folder,
            num_images=len(image_files),
        )

        # Check ALL image sizes
        sizes_found = {}

        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape[:2]
                size_key = (w, h)
                sizes_found[size_key] = sizes_found.get(size_key, 0) + 1

        stats.sizes_found = sizes_found

        if sizes_found:
            # Most common size becomes the "primary" size
            primary_size = max(sizes_found.keys(), key=lambda k: sizes_found[k])
            stats.image_width, stats.image_height = primary_size

            # Check consistency
            stats.size_consistent = (len(sizes_found) == 1)

        # Add focus metrics + metadata from CSV if available
        lookup_key = csv_key if csv_key is not None else folder.name
        if csv_data and lookup_key in csv_data:
            folder_records = csv_data[lookup_key]
            if camera_filter:
                folder_records = [r for r in folder_records if r.get('camera') == camera_filter]
            if folder_records:
                laplacians = [r['laplacian_var'] for r in folder_records
                              if 'laplacian_var' in r and r['laplacian_var'] ==r['laplacian_var']]
                tenengrads = [r['tenengrad'] for r in folder_records
                              if 'tenengrad' in r and r['tenengrad'] ==r['tenengrad']]

                if laplacians:
                    stats.has_focus_metrics = True
                    stats.mean_laplacian = sum(laplacians) / len(laplacians)
                    stats.min_laplacian = min(laplacians)
                    stats.max_laplacian = max(laplacians)

                if tenengrads:
                    stats.mean_tenengrad = sum(tenengrads) / len(tenengrads)

                # Scale, native blur, diameter, camera
                scales = [float(r['scale_px_per_mm']) for r in folder_records
                          if 'scale_px_per_mm' in r and r['scale_px_per_mm'] ==
                          r['scale_px_per_mm']]
                blurs = [
                    float(r['native_blur_sigma']) for r in folder_records
                    if 'native_blur_sigma' in r and r['native_blur_sigma'] ==
                    r['native_blur_sigma']]
                diams = [float(r['diameter_px']) for r in folder_records
                         if 'diameter_px' in r and r['diameter_px'] == r['diameter_px']]
                cameras = [r['camera'] for r in folder_records if 'camera' in r and r['camera']]

                if scales:
                    stats.has_csv_metadata = True
                    stats.mean_scale_px_per_mm = sum(scales) / len(scales)
                    stats.min_scale_px_per_mm = min(scales)
                    stats.max_scale_px_per_mm = max(scales)
                if blurs:
                    stats.has_csv_metadata = True
                    stats.mean_native_blur = sum(blurs) / len(blurs)
                    stats.min_native_blur = min(blurs)
                    stats.max_native_blur = max(blurs)
                if diams:
                    stats.has_csv_metadata = True
                    stats.mean_diameter_px = sum(diams) / len(diams)
                    stats.min_diameter_px = min(diams)
                    stats.max_diameter_px = max(diams)
                if cameras:
                    # Take most common camera label
                    stats.camera = max(set(cameras), key=cameras.count)

        # Log result
        if log_callback:
            size_info = f"{stats.image_width}×{stats.image_height}"
            if not stats.size_consistent:
                size_str = ", ".join(f"{w}×{h}({n})" for (w, h), n in sizes_found.items())
                size_info = f"MIXED: {size_str}"

            blur_info = ""
            if stats.has_csv_metadata and stats.mean_native_blur > 0:
                blur_info = f", σ_native=[{stats.min_native_blur:.3f}–{stats.max_native_blur:.3f}] px"

            prefix = "⚠ " if not stats.size_consistent else "  "
            log_callback(
                f"{prefix}{folder.name}: {stats.num_images} images, {size_info}{blur_info}")

        return stats

    def check_cross_folder_consistency(self, stats_dict: Dict[str, FolderStats]) -> Tuple[bool, str]:
        """Check if all folders have the same image size."""
        if not stats_dict:
            return True, "No folders to check"

        # First check if any folder has internal inconsistency
        inconsistent_folders = [name for name, s in stats_dict.items() if not s.size_consistent]

        # Then check cross-folder consistency
        sizes = {}
        for folder_name, stats in stats_dict.items():
            if stats.image_width > 0 and stats.image_height > 0:
                size = (stats.image_width, stats.image_height)
                if size not in sizes:
                    sizes[size] = []
                sizes[size].append(folder_name)

        msg_parts = []
        all_consistent = True

        if inconsistent_folders:
            all_consistent = False
            msg_parts.append(f"⚠ Mixed sizes WITHIN folders: {', '.join(inconsistent_folders)}")

        if len(sizes) == 0:
            msg_parts.append("No image sizes determined")
        elif len(sizes) == 1:
            size = list(sizes.keys())[0]
            msg_parts.append(f"All folders: {size[0]}×{size[1]}")
        else:
            all_consistent = False
            msg_parts.append("Mixed sizes ACROSS folders:")
            for size, folders in sizes.items():
                folder_list = ', '.join(folders[:3])
                if len(folders) > 3:
                    folder_list += f" (+{len(folders)-3} more)"
                msg_parts.append(f"  {size[0]}×{size[1]}: {folder_list}")

        return all_consistent, "\n".join(msg_parts)


# =============================================================================
# Training GUI
# =============================================================================
class TrainingGUI:
    """Main training GUI application."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Defocus Estimation - Training Suite")
        self.root.geometry("1100x800")

        # State
        self.sharp_crops_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.scanner = SharpCropsScanner()
        self.folder_stats: Dict[str, FolderStats] = {}
        self.folder_configs: Dict[str, OpticalConfig] = {}
        self.selected_folder: Optional[str] = None
        self.global_config: Optional[Dict[str, Any]] = None
        self.all_sizes_consistent: bool = True

        # Training state
        self.training_thread: Optional[threading.Thread] = None
        self.stop_training = False

        # Message queue for thread communication
        self.msg_queue = queue.Queue()

        # Build UI
        self._create_ui()

        # Start message processor
        self._process_messages()

    def _create_ui(self):
        """Create the main UI."""
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill='both', expand=True)

        # Notebook with tabs (paths are now in individual tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True, pady=(10, 0))

        # Tab 1: Scan & Configure
        self.tab_config = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_config, text="1. Scan & Configure")
        self._create_config_tab()

        # Tab 2: Generate
        self.tab_generate = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_generate, text="2. Generate")
        self._create_generate_tab()

        # Tab 3: Train
        self.tab_train = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_train, text="3. Train")
        self._create_train_tab()

        # Tab 4: Validation
        self.tab_validation = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_validation, text="4. Validation")
        self._create_validation_tab()

        # Tab 5: Inference (with scrolling)
        self.tab_inference = ttk.Frame(self.notebook, padding=0)
        self.notebook.add(self.tab_inference, text="5. Inference")
        self._create_inference_tab_scrollable()

        # Bottom: Log output
        self._create_log_section(main_frame)

    def _create_path_section(self, parent):
        """Create path selection section."""
        frame = ttk.LabelFrame(parent, text="Sharp Crops Path", padding=10)
        frame.pack(fill='x', pady=(0, 5))

        # Sharp Crops (from preprocessing)
        row1 = ttk.Frame(frame)
        row1.pack(fill='x', pady=2)
        ttk.Label(row1, text="Sharp Crops:", width=15).pack(side='left')
        self.sharp_crops_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self.sharp_crops_var, width=50).pack(side='left', padx=5)
        ttk.Button(row1, text="Browse", command=self._browse_sharp_crops).pack(side='left')

        # Camera filter
        ttk.Label(row1, text="Camera:").pack(side='left', padx=(10, 2))
        self.camera_filter_var = tk.StringVar(value="all")
        camera_combo = ttk.Combobox(row1, textvariable=self.camera_filter_var, width=6,
                                    values=["all", "g", "m", "v"], state="readonly")
        camera_combo.pack(side='left')

        # Note: Output directory is in Tab 2 (Generate)

    def _create_config_tab(self):
        """Create scan & configure tab."""
        # Add path section at the top
        self._create_path_section(self.tab_config)

        # Left side: Folder list
        left_frame = ttk.Frame(self.tab_config)
        left_frame.pack(side='left', fill='both', expand=True)

        ttk.Label(left_frame, text="Sharp Crop Folders:").pack(anchor='w')

        # Scan button
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="Load from CSVs",
                   command=self._scan_sharp_crops).pack(side='left')

        # Folder listbox with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill='both', expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')

        self.folder_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, width=30)
        self.folder_listbox.pack(side='left', fill='both', expand=True)
        self.folder_listbox.bind('<<ListboxSelect>>', self._on_folder_select)
        scrollbar.config(command=self.folder_listbox.yview)

        # Delete button
        ttk.Button(
            left_frame, text="Remove Selected", command=self._remove_selected_folder).pack(
            anchor='w', pady=(5, 0))

        # Summary label
        self.summary_var = tk.StringVar(value="No folders loaded")
        ttk.Label(left_frame, textvariable=self.summary_var,
                  wraplength=250).pack(anchor='w', pady=5)

        # Right side: Configuration
        right_frame = ttk.LabelFrame(self.tab_config, text="Configuration", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))

        self._create_config_fields(right_frame)

    def _create_config_fields(self, parent):
        """Create optical configuration input fields."""
        # Folder stats (read-only)
        info_frame = ttk.LabelFrame(parent, text="Folder Statistics", padding=5)
        info_frame.pack(fill='x', pady=5)

        self.folder_stats_text = tk.Text(info_frame, height=10, width=48, state='disabled')
        self.folder_stats_text.pack(fill='x')

        # Training Mode selection (NEW - before lens parameters)
        mode_frame = ttk.LabelFrame(parent, text="Training Mode", padding=5)
        mode_frame.pack(fill='x', pady=5)

        self.training_mode_var = tk.StringVar(value="optical")

        mode_row = ttk.Frame(mode_frame)
        mode_row.pack(fill='x', pady=2)

        ttk.Radiobutton(
            mode_row,
            text="Optical Formula (CoC-based, requires known optical parameters)",
            variable=self.training_mode_var,
            value="optical",
            command=self._on_training_mode_change
        ).pack(side='left', padx=10)

        ttk.Radiobutton(
            mode_row,
            text="Direct Calibration (linear blur, uses calibration data)",
            variable=self.training_mode_var,
            value="direct",
            command=self._on_training_mode_change
        ).pack(side='left', padx=10)

        # Mode description label
        self.mode_desc_var = tk.StringVar(
            value="Uses optical formula (Wang et al.) to generate blur from known optical parameters")
        ttk.Label(mode_frame, textvariable=self.mode_desc_var,
                  font=('', 8), foreground='gray', wraplength=500).pack(anchor='w', pady=(5, 0))

        # Lens parameters (user input) with mode toggle
        self.lens_frame = ttk.LabelFrame(parent, text="Lens Parameters (Enter Manually)", padding=5)
        self.lens_frame.pack(fill='x', pady=5)

        # Mode toggle row - Global vs Per Subfolder
        self.config_mode_row = ttk.Frame(self.lens_frame)
        self.config_mode_row.pack(fill='x', pady=(0, 8))

        ttk.Label(self.config_mode_row, text="Apply to:", width=12).pack(side='left')
        self.config_mode_var = tk.StringVar(value="global")

        self.global_radio = ttk.Radiobutton(
            self.config_mode_row,
            text="All Folders (Global)",
            variable=self.config_mode_var,
            value="global",
            command=self._on_mode_change
        )
        self.global_radio.pack(side='left', padx=(0, 10))

        self.per_folder_radio = ttk.Radiobutton(
            self.config_mode_row,
            text="Per Subfolder",
            variable=self.config_mode_var,
            value="per_folder",
            command=self._on_mode_change
        )
        self.per_folder_radio.pack(side='left')

        # Separator
        self.lens_separator = ttk.Separator(self.lens_frame, orient='horizontal')
        self.lens_separator.pack(fill='x', pady=5)

        # Optical Parameters frame (wrapped for show/hide)
        self.optical_params_frame = ttk.Frame(self.lens_frame)
        self.optical_params_frame.pack(fill='x', pady=0)

        # Focal length
        row1 = ttk.Frame(self.optical_params_frame)
        row1.pack(fill='x', pady=2)
        ttk.Label(row1, text="Focal Length (mm):", width=20).pack(side='left')
        self.focal_length_var = tk.StringVar(value="200.0")
        ttk.Entry(row1, textvariable=self.focal_length_var, width=15).pack(side='left')

        # F-number
        row2 = ttk.Frame(self.optical_params_frame)
        row2.pack(fill='x', pady=2)
        ttk.Label(row2, text="F-number (f/):", width=20).pack(side='left')
        self.f_number_var = tk.StringVar(value="4.0")
        ttk.Entry(row2, textvariable=self.f_number_var, width=15).pack(side='left')

        # Focus distance
        row3 = ttk.Frame(self.optical_params_frame)
        row3.pack(fill='x', pady=2)
        ttk.Label(row3, text="Focus Distance (mm):", width=20).pack(side='left')
        self.focus_distance_var = tk.StringVar(value="400.0")
        ttk.Entry(row3, textvariable=self.focus_distance_var, width=15).pack(side='left')

        # Pixel size (training camera)
        row4 = ttk.Frame(self.optical_params_frame)
        row4.pack(fill='x', pady=2)
        ttk.Label(row4, text="Pixel Size (mm):", width=20).pack(side='left')
        self.pixel_size_var = tk.StringVar(value="0.020")
        ttk.Entry(row4, textvariable=self.pixel_size_var, width=12).pack(side='left')
        ttk.Button(
            row4, text="?", width=2, command=self._lookup_training_pixel_size).pack(
            side='left', padx=2)

        # Training crop size (original size before resize to 128)
        row4b = ttk.Frame(self.optical_params_frame)
        row4b.pack(fill='x', pady=2)
        ttk.Label(row4b, text="Crop Size (px):", width=20).pack(side='left')
        self.training_crop_size_var = tk.StringVar(value="299")
        ttk.Entry(row4b, textvariable=self.training_crop_size_var, width=12).pack(side='left')
        ttk.Label(row4b, text="(original crop before 128px resize)",
                  font=('', 7)).pack(side='left', padx=5)

        # Direct Calibration Parameters frame (NEW - for direct mode)
        self.direct_params_frame = ttk.Frame(self.lens_frame)
        # Not packed initially - shown when mode changes to direct

        # Browse for calibration file
        row_browse = ttk.Frame(self.direct_params_frame)
        row_browse.pack(fill='x', pady=2)
        ttk.Label(row_browse, text="Calibration File:", width=20).pack(side='left')
        self.direct_calib_path_var = tk.StringVar()
        self.direct_calib_entry = ttk.Entry(row_browse, textvariable=self.direct_calib_path_var,
                                            width=40, state='readonly')
        self.direct_calib_entry.pack(side='left', padx=5)
        ttk.Button(
            row_browse, text="Browse...", command=self._browse_direct_calibration).pack(
            side='left')

        # Read-only displays for loaded values (USER CONSTRAINT: no manual editing)
        row_rho = ttk.Frame(self.direct_params_frame)
        row_rho.pack(fill='x', pady=2)
        ttk.Label(row_rho, text="ρ_direct (px/mm):", width=20).pack(side='left')
        self.rho_direct_var = tk.StringVar(value="")
        ttk.Entry(row_rho, textvariable=self.rho_direct_var,
                  width=12, state='readonly').pack(side='left')

        row_sigma = ttk.Frame(self.direct_params_frame)
        row_sigma.pack(fill='x', pady=2)
        ttk.Label(row_sigma, text="σ₀ (px):", width=20).pack(side='left')
        self.sigma_0_var = tk.StringVar(value="")
        ttk.Entry(row_sigma, textvariable=self.sigma_0_var,
                  width=12, state='readonly').pack(side='left')

        # Info label
        ttk.Label(
            self.direct_params_frame,
            text="Load calibration file from direct calibration workflow to populate these values.",
            font=('', 8),
            foreground='gray', wraplength=500).pack(
            anchor='w', pady=5)

        # Calibration reference section (loaded from calibration file)
        self.calib_ref_frame = ttk.LabelFrame(
            parent, text="Calibration Reference (from calibration file)", padding=5)
        self.calib_ref_frame.pack(fill='x', pady=5)

        row_calib1 = ttk.Frame(self.calib_ref_frame)
        row_calib1.pack(fill='x', pady=2)
        ttk.Label(row_calib1, text="Calib Pixel Size (mm):", width=20).pack(side='left')
        self.calib_pixel_size_var = tk.StringVar(value="")
        calib_px_entry = ttk.Entry(
            row_calib1, textvariable=self.calib_pixel_size_var, width=12, state='readonly')
        calib_px_entry.pack(side='left')
        ttk.Label(row_calib1, text="(auto-loaded)", font=('', 7)).pack(side='left', padx=5)

        row_calib2 = ttk.Frame(self.calib_ref_frame)
        row_calib2.pack(fill='x', pady=2)
        ttk.Label(row_calib2, text="Calib Resolution (px):", width=20).pack(side='left')
        self.calib_reference_resolution_var = tk.StringVar(value="")
        calib_res_entry = ttk.Entry(
            row_calib2, textvariable=self.calib_reference_resolution_var, width=12,
            state='readonly')
        calib_res_entry.pack(side='left')
        ttk.Label(row_calib2, text="(auto-loaded)", font=('', 7)).pack(side='left', padx=5)

        row_calib3 = ttk.Frame(self.calib_ref_frame)
        row_calib3.pack(fill='x', pady=2)
        ttk.Label(row_calib3, text="Calib Scale (px/mm):", width=20).pack(side='left')
        self.calib_scale_px_per_mm_var = tk.StringVar(value="")
        calib_scale_entry = ttk.Entry(
            row_calib3, textvariable=self.calib_scale_px_per_mm_var, width=12, state='readonly')
        calib_scale_entry.pack(side='left')
        ttk.Label(row_calib3, text="(auto-loaded)", font=('', 7)).pack(side='left', padx=5)

        # Training parameters
        train_frame = ttk.LabelFrame(parent, text="Training Parameters", padding=5)
        train_frame.pack(fill='x', pady=5)

        # Defocus range
        row5 = ttk.Frame(train_frame)
        row5.pack(fill='x', pady=2)
        ttk.Label(row5, text="Defocus Range (mm):", width=20).pack(side='left')
        self.defocus_min_var = tk.StringVar(value="-12.0")
        ttk.Entry(row5, textvariable=self.defocus_min_var, width=8).pack(side='left')
        ttk.Label(row5, text=" to ").pack(side='left')
        self.defocus_max_var = tk.StringVar(value="12.0")
        ttk.Entry(row5, textvariable=self.defocus_max_var, width=8).pack(side='left')

        # Rho
        self.rho_row = ttk.Frame(train_frame)
        self.rho_row.pack(fill='x', pady=2)
        ttk.Label(self.rho_row, text="ρ (blur constant):", width=20).pack(side='left')
        self.rho_var = tk.StringVar(value="1.0")
        ttk.Entry(self.rho_row, textvariable=self.rho_var, width=15).pack(side='left')

        # Calculated values display
        self.calc_values_frame = ttk.LabelFrame(parent, text="Calculated Values", padding=5)
        self.calc_values_frame.pack(fill='x', pady=5)

        self.calculated_text = tk.Text(self.calc_values_frame, height=3, width=40, state='disabled')
        self.calculated_text.pack(fill='x')

        # Buttons - changes based on mode
        self.config_btn_frame = ttk.Frame(parent)
        self.config_btn_frame.pack(fill='x', pady=10)

        self.load_calib_btn = ttk.Button(
            self.config_btn_frame, text="Load from Calibration", command=self._load_from_calibration)
        self.load_calib_btn.pack(side='left', padx=5)
        self.calculate_btn = ttk.Button(self.config_btn_frame, text="Calculate",
                                         command=self._update_calculated)
        self.calculate_btn.pack(side='left', padx=5)

        # Save button - text changes based on mode
        self.save_config_btn = ttk.Button(
            self.config_btn_frame, text="Save Config (Global)", command=self._save_config)
        self.save_config_btn.pack(side='left', padx=5)

    def _create_generate_tab(self):
        """Create synthetic data generation tab."""

        # Output root + dataset name
        output_dir_frame = ttk.LabelFrame(self.tab_generate, text="Output Location", padding=10)
        output_dir_frame.pack(fill='x', padx=5, pady=(5, 10))

        row = ttk.Frame(output_dir_frame)
        row.pack(fill='x', pady=2)
        ttk.Label(row, text="Training output root:", width=20).pack(side='left')
        self.output_dir_var = tk.StringVar(value=str(Path.cwd() / "training_output"))
        ttk.Entry(row, textvariable=self.output_dir_var, width=65).pack(side='left', padx=5)
        ttk.Button(row, text="Browse", command=self._browse_output_dir).pack(side='left')

        row_name = ttk.Frame(output_dir_frame)
        row_name.pack(fill='x', pady=2)
        ttk.Label(row_name, text="Dataset name (optional):", width=20).pack(side='left')
        self.dataset_name_var = tk.StringVar(value="")
        ttk.Entry(row_name, textvariable=self.dataset_name_var, width=65).pack(side='left', padx=5)
        ttk.Label(output_dir_frame,
                  text="Datasets are saved as <root>/datasets/<timestamp>_<name>/",
                  font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', pady=(2, 0))

        # Create two-column layout
        main_container = ttk.Frame(self.tab_generate)
        main_container.pack(fill='both', expand=True, padx=5, pady=5)

        # Left column for settings
        left_column = ttk.Frame(main_container)
        left_column.pack(side='left', fill='both', expand=False, padx=(0, 5))

        # Right column for calculator
        right_column = ttk.Frame(main_container)
        right_column.pack(side='left', fill='both', expand=True, padx=(5, 0))

        # Settings frame (in left column)
        settings_frame = ttk.LabelFrame(left_column, text="Generation Settings", padding=10)
        settings_frame.pack(fill='x', pady=5)

        # Number of synthetic samples
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill='x', pady=2)
        ttk.Label(row1, text="Synthetic Samples:", width=20).pack(side='left')
        self.num_samples_var = tk.StringVar(value="50000")
        ttk.Entry(row1, textvariable=self.num_samples_var, width=15).pack(side='left')
        ttk.Label(row1, text="(more = better model, slower)").pack(side='left', padx=5)

        # Training crop size
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill='x', pady=2)
        ttk.Label(row2, text="Output Size:", width=20).pack(side='left')
        self.train_size_var = tk.StringVar(value="128")
        ttk.Entry(row2, textvariable=self.train_size_var, width=15).pack(side='left')
        ttk.Label(row2, text="px (smaller = faster training)").pack(side='left', padx=5)

        # Blur Distribution
        coc_frame = ttk.LabelFrame(settings_frame, text="CoC Sampling Distribution", padding=5)
        coc_frame.pack(fill='x', pady=(10, 5))
        self.coc_frame = coc_frame  # stored for label updates on mode change

        # Distribution type
        row3 = ttk.Frame(coc_frame)
        row3.pack(fill='x', pady=2)
        ttk.Label(row3, text="Distribution:", width=20).pack(side='left')
        self.blur_distribution_var = tk.StringVar(value="uniform")
        uniform_rb = ttk.Radiobutton(
            row3, text="Uniform", variable=self.blur_distribution_var, value="uniform",
            command=self._on_blur_distribution_change)
        uniform_rb.pack(side='left', padx=5)
        weighted_rb = ttk.Radiobutton(
            row3, text="Weighted", variable=self.blur_distribution_var, value="weighted",
            command=self._on_blur_distribution_change)
        weighted_rb.pack(side='left', padx=5)

        # Beta parameters (only shown for weighted)
        self.beta_params_frame = ttk.Frame(coc_frame)
        self.beta_params_frame.pack(fill='x', pady=2)

        ttk.Label(self.beta_params_frame, text="Beta α:", width=10).pack(side='left')
        self.beta_alpha_var = tk.StringVar(value="2.0")
        self.beta_alpha_entry = ttk.Entry(
            self.beta_params_frame, textvariable=self.beta_alpha_var, width=8)
        self.beta_alpha_entry.pack(side='left', padx=(0, 10))

        ttk.Label(self.beta_params_frame, text="Beta β:", width=10).pack(side='left')
        self.beta_beta_var = tk.StringVar(value="5.0")
        self.beta_beta_entry = ttk.Entry(
            self.beta_params_frame, textvariable=self.beta_beta_var, width=8)
        self.beta_beta_entry.pack(side='left', padx=(0, 10))

        self.beta_hint_label = ttk.Label(
            self.beta_params_frame, text="(α<β favors small CoC)", font=('', 8),
            foreground='gray')
        self.beta_hint_label.pack(side='left')

        # Initially disable beta params
        self._on_blur_distribution_change()

        # Info text (in left column)
        info_frame = ttk.LabelFrame(left_column, text="Info", padding=10)
        info_frame.pack(fill='x', pady=5)

        info_text = """Generation creates synthetic training data by:
1. Taking your sharp crop images
2. Applying known amounts of blur (based on config)
3. Sampling blur values from chosen distribution (uniform/weighted)
4. Saving triplets: (blurred, sharp, blur_map)

Outputs (saved to synthetic_data/):
  • blur/ - Blurred images
  • sharp/ - Ground truth sharp images
  • blur_map/ - Blur maps (normalized)
  • metadata.csv - Blur values, defocus distances
  • blur_distribution.png - Histogram visualization
  • config.yaml - Generation parameters

This gives the model examples with known ground truth to learn from."""

        ttk.Label(info_frame, text=info_text, justify='left').pack(anchor='w')

        # Progress (in left column)
        progress_frame = ttk.LabelFrame(left_column, text="Progress", padding=10)
        progress_frame.pack(fill='x', pady=5)

        self.gen_progress_var = tk.DoubleVar(value=0)
        self.gen_progress_bar = ttk.Progressbar(
            progress_frame, variable=self.gen_progress_var, maximum=100)
        self.gen_progress_bar.pack(fill='x', pady=5)

        self.gen_status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.gen_status_var).pack(anchor='w')

        # Action buttons (in left column)
        btn_frame = ttk.Frame(left_column)
        btn_frame.pack(fill='x', pady=10)

        self.generate_btn = ttk.Button(
            btn_frame,
            text="Generate Synthetic Data",
            command=self._generate_data
        )
        self.generate_btn.pack(side='left', padx=5)

        # Minimum Blur Filter section (RIGHT column, above calculator)
        min_blur_frame = ttk.LabelFrame(right_column, text="Minimum CoC Filter", padding=10)
        min_blur_frame.pack(fill='x', padx=5, pady=(0, 10))
        self.min_blur_frame = min_blur_frame  # stored for label updates on mode change

        # Checkbox for enabling minimum blur filter
        self.min_blur_enabled_var = tk.BooleanVar(value=False)
        self.min_blur_check = ttk.Checkbutton(
            min_blur_frame,
            text="Apply minimum CoC threshold",
            variable=self.min_blur_enabled_var,
            command=self._on_min_blur_toggle
        )
        self.min_blur_check.grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 5))

        # Input for minimum blur value
        self.min_blur_label = ttk.Label(min_blur_frame, text="Min CoC (px):")
        self.min_blur_label.grid(row=1, column=0, sticky='w', padx=(20, 5))
        self.min_blur_value_var = tk.StringVar(value="0.5")
        min_blur_entry = ttk.Entry(min_blur_frame, textvariable=self.min_blur_value_var, width=10)
        min_blur_entry.grid(row=1, column=1, sticky='w')

        # Initially disable the entry field
        self.min_blur_entry = min_blur_entry
        self.min_blur_entry.config(state='disabled')

        # Info text
        info_label = ttk.Label(
            min_blur_frame,
            text="ℹ️  Value is in calibration-camera pixels. Bins and intervals shift to [min, max]",
            font=('TkDefaultFont', 8),
            foreground='gray')
        info_label.grid(row=2, column=0, columnspan=2, sticky='w', pady=(5, 0))

        # Preview button to visualize blur at different levels
        preview_btn = ttk.Button(
            min_blur_frame,
            text="Preview Blur Levels",
            command=self._preview_blur_levels
        )
        preview_btn.grid(row=3, column=0, columnspan=2, sticky='w', padx=(20, 5), pady=(10, 0))

        # Blur trace metadata option
        trace_frame = ttk.LabelFrame(right_column, text="Diagnostic Options", padding=5)
        trace_frame.pack(fill='x', padx=5, pady=(0, 10))
        self.save_blur_trace_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            trace_frame,
            text="Save blur trace metadata",
            variable=self.save_blur_trace_var,
        ).pack(anchor='w')

        # ERF validation option
        self.erf_validation_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            trace_frame,
            text="Run ERF validation after generation",
            variable=self.erf_validation_var,
            command=self._on_erf_validation_toggle,
        ).pack(anchor='w', pady=(5, 0))

        erf_count_row = ttk.Frame(trace_frame)
        erf_count_row.pack(anchor='w', padx=(20, 0), pady=(2, 0))
        ttk.Label(erf_count_row, text="Validation samples:").pack(side='left')
        self.erf_validation_count_var = tk.StringVar(value=self.num_samples_var.get())
        self.erf_validation_count_entry = ttk.Entry(
            erf_count_row, textvariable=self.erf_validation_count_var, width=10
        )
        self.erf_validation_count_entry.pack(side='left', padx=(5, 0))
        self.erf_validation_count_entry.config(state='disabled')

        # Keep ERF count in sync with synthetic samples when user hasn't overridden
        self._erf_count_user_modified = False

        def _on_num_samples_change(*_args):
            if not self._erf_count_user_modified:
                self.erf_validation_count_var.set(self.num_samples_var.get())
        self.num_samples_var.trace_add('write', _on_num_samples_change)

        def _on_erf_count_edit(*_args):
            # Mark as user-modified if it differs from num_samples
            if self.erf_validation_count_var.get() != self.num_samples_var.get():
                self._erf_count_user_modified = True
        self.erf_validation_count_var.trace_add('write', _on_erf_count_edit)

        # Create two calculator sections side by side
        calc_container = ttk.Frame(right_column)
        calc_container.pack(fill='both', expand=False, pady=5)
        self.calc_container = calc_container  # stored for label updates on mode change

        # Left: Forward Calculator (Beta → Distribution)
        forward_frame = ttk.LabelFrame(
            calc_container, text="Forward: Beta → Distribution", padding=10)
        forward_frame.pack(side='left', fill='both', expand=False, padx=(0, 5))

        # Input section
        input_section = ttk.Frame(forward_frame)
        input_section.pack(fill='x', pady=5)

        ttk.Label(input_section, text="Beta α:", width=8).pack(side='left', padx=(0, 5))
        self.calc_alpha_var = tk.StringVar(value="2.0")
        ttk.Entry(input_section, textvariable=self.calc_alpha_var,
                  width=10).pack(side='left', padx=(0, 15))

        ttk.Label(input_section, text="Beta β:", width=8).pack(side='left', padx=(0, 5))
        self.calc_beta_var = tk.StringVar(value="5.0")
        ttk.Entry(input_section, textvariable=self.calc_beta_var,
                  width=10).pack(side='left', padx=(0, 15))

        ttk.Button(input_section, text="Calculate",
                   command=self._calculate_beta_distribution).pack(side='left')

        # Results section
        results_frame = ttk.Frame(forward_frame)
        results_frame.pack(fill='both', expand=False, pady=(10, 0))

        self.beta_calc_results = tk.Text(results_frame, height=15,
                                         width=50, state='disabled', font=('Courier', 9))
        self.beta_calc_results.pack()

        # Set initial placeholder message
        self.beta_calc_results.config(state='normal')
        self.beta_calc_results.insert('1.0',
                                      "Press 'Calculate' to compute the CoC\n"
                                      "distribution for your Beta parameters.\n\n"
                                      "Results will be based on the optical\n"
                                      "configuration saved in Tab 1.\n\n"
                                      "The distribution will show:\n"
                                      "  • CoC range (0 to max CoC in px)\n"
                                      "  • Mean, Median, Std Dev\n"
                                      "  • Distribution across 4 intervals\n"
                                      "  • Visual bar charts"
                                      )
        self.beta_calc_results.config(state='disabled')

        # Right: Reverse Calculator (Distribution → Beta)
        reverse_frame = ttk.LabelFrame(
            calc_container, text="Reverse: Target Distribution → Beta", padding=10)
        reverse_frame.pack(side='left', fill='both', expand=False, padx=(5, 0))

        # Target distribution inputs
        ttk.Label(reverse_frame, text="Enter desired % in each interval:").pack(pady=(0, 10))

        self.reverse_interval_vars = []
        for i in range(4):
            row = ttk.Frame(reverse_frame)
            row.pack(fill='x', pady=2)
            ttk.Label(row, text=f"Interval {i+1}:", width=12).pack(side='left', padx=(0, 5))
            var = tk.StringVar(value="25.0")
            ttk.Entry(row, textvariable=var, width=10).pack(side='left', padx=(0, 5))
            ttk.Label(row, text="%").pack(side='left')
            self.reverse_interval_vars.append(var)

        ttk.Button(reverse_frame, text="Find Beta Params",
                   command=self._reverse_calculate_beta).pack(pady=(10, 5))

        # Reverse results
        reverse_results_frame = ttk.Frame(reverse_frame)
        reverse_results_frame.pack(fill='both', expand=False, pady=(10, 0))

        self.reverse_calc_results = tk.Text(
            reverse_results_frame, height=8, width=40, state='disabled', font=('Courier', 9))
        self.reverse_calc_results.pack()

    def _create_train_tab(self):
        """Create model training tab with mode selection."""

        # Mode selection frame
        mode_frame = ttk.LabelFrame(self.tab_train, text="Select Training Mode", padding=10)
        mode_frame.pack(fill='x', pady=5)

        self.train_mode_var = tk.StringVar(value="")

        mode_btn_frame = ttk.Frame(mode_frame)
        mode_btn_frame.pack(fill='x')

        self.mode_full_btn = ttk.Button(
            mode_btn_frame, text="Train Full Model",
            command=lambda: self._select_train_mode("full")
        )
        self.mode_full_btn.pack(side='left', padx=5, pady=5)

        self.mode_dme_btn = ttk.Button(
            mode_btn_frame, text="Train DME Only",
            command=lambda: self._select_train_mode("dme")
        )
        self.mode_dme_btn.pack(side='left', padx=5, pady=5)

        # Checkpoint selection (right side)
        checkpoint_frame = ttk.Frame(mode_btn_frame)
        checkpoint_frame.pack(side='right', padx=5)

        ttk.Label(
            checkpoint_frame, text="Resume from:", font=('', 9)).pack(
            side='left', padx=(0, 5))

        self.checkpoint_path_var = tk.StringVar(value="")
        self.checkpoint_display_var = tk.StringVar(value="Auto-detect")

        # Add trace to update validation split state when checkpoint changes
        self.checkpoint_path_var.trace_add('write', lambda *args: self._update_val_split_state())

        checkpoint_entry = ttk.Entry(
            checkpoint_frame,
            textvariable=self.checkpoint_display_var,
            width=20,
            state='readonly'
        )
        checkpoint_entry.pack(side='left', padx=(0, 5))

        ttk.Button(
            checkpoint_frame,
            text="Browse...",
            command=self._browse_checkpoint,
            width=10
        ).pack(side='left', padx=(0, 5))

        ttk.Button(
            checkpoint_frame,
            text="Clear",
            command=self._clear_checkpoint,
            width=6
        ).pack(side='left')

        # Info row: mode description (left) and scan checkpoint (right) - all on same line
        info_row = ttk.Frame(mode_frame)
        info_row.pack(fill='x', pady=(2, 0))

        # Left side: Mode description and checkpoint info stacked
        left_info = ttk.Frame(info_row)
        left_info.pack(side='left', fill='x', expand=True)

        self.mode_desc_var = tk.StringVar(value="Select a training mode above")
        ttk.Label(left_info, textvariable=self.mode_desc_var,
                  foreground='gray', font=('', 8)).pack(anchor='w')

        self.checkpoint_info_var = tk.StringVar(value="")
        self.checkpoint_info_label = ttk.Label(
            left_info, textvariable=self.checkpoint_info_var, foreground='blue', font=('', 8))
        self.checkpoint_info_label.pack(anchor='w')

        # Right side: Scan checkpoint button and metrics
        scan_inner = ttk.Frame(info_row)
        scan_inner.pack(side='right', padx=5)

        self.checkpoint_metrics_var = tk.StringVar(value="")
        ttk.Label(scan_inner, textvariable=self.checkpoint_metrics_var,
                  foreground='gray', font=('', 8)).pack(side='left', padx=(0, 5))

        ttk.Button(
            scan_inner,
            text="Scan Checkpoint",
            command=self._scan_checkpoint_metrics,
            width=15
        ).pack(side='left')

        # Dataset & run selection
        self.dataset_frame = ttk.LabelFrame(self.tab_train, text="Dataset & Run", padding=10)
        self.dataset_frame.pack(fill='x', pady=5)

        ds_row = ttk.Frame(self.dataset_frame)
        ds_row.pack(fill='x', pady=2)
        ttk.Label(ds_row, text="Dataset:", width=14).pack(side='left')
        self.dataset_path_var = tk.StringVar(value="")
        self.dataset_combo = ttk.Combobox(ds_row, textvariable=self.dataset_path_var,
                                           width=70, state='readonly')
        self.dataset_combo.pack(side='left', padx=5)
        self.dataset_combo.bind('<<ComboboxSelected>>', lambda _e: self._on_dataset_select())
        ttk.Button(ds_row, text="Refresh", command=self._refresh_datasets, width=8).pack(side='left')
        ttk.Button(ds_row, text="Browse...", command=self._browse_dataset, width=10).pack(side='left',
                                                                                             padx=2)

        self.dataset_info_var = tk.StringVar(value="No dataset selected")
        ttk.Label(self.dataset_frame, textvariable=self.dataset_info_var,
                  font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', pady=(2, 4))

        run_row = ttk.Frame(self.dataset_frame)
        run_row.pack(fill='x', pady=2)
        ttk.Label(run_row, text="Run name (optional):", width=20).pack(side='left')
        self.run_name_var = tk.StringVar(value="")
        ttk.Entry(run_row, textvariable=self.run_name_var, width=40).pack(side='left', padx=5)
        ttk.Label(self.dataset_frame,
                  text="Run output: <root>/runs/<timestamp>_<name>/",
                  font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', pady=(2, 0))

        # Settings frame (initially disabled)
        self.settings_frame = ttk.LabelFrame(self.tab_train, text="Settings", padding=10)
        self.settings_frame.pack(fill='x', pady=5)

        # Create two columns
        left_col = ttk.Frame(self.settings_frame)
        left_col.pack(side='left', fill='both', expand=True)

        right_col = ttk.Frame(self.settings_frame)
        right_col.pack(side='left', fill='both', expand=True, padx=(20, 0))

        # DME settings (left column)
        self.dme_settings_frame = ttk.LabelFrame(left_col, text="DME Settings", padding=5)
        self.dme_settings_frame.pack(fill='x', pady=2)

        row1 = ttk.Frame(self.dme_settings_frame)
        row1.pack(fill='x', pady=2)
        ttk.Label(row1, text="Epochs to Train:", width=14).pack(side='left')
        self.epochs_dme_var = tk.StringVar(value="50")
        self.epochs_dme_entry = ttk.Entry(row1, textvariable=self.epochs_dme_var, width=10)
        self.epochs_dme_entry.pack(side='left')

        row2 = ttk.Frame(self.dme_settings_frame)
        row2.pack(fill='x', pady=2)
        ttk.Label(row2, text="Batch Size:", width=12).pack(side='left')
        self.batch_size_var = tk.StringVar(value="128")
        self.batch_dme_entry = ttk.Entry(row2, textvariable=self.batch_size_var, width=10)
        self.batch_dme_entry.pack(side='left')

        # Learning rate (right column)
        self.lr_frame = ttk.LabelFrame(right_col, text="Learning Rate", padding=5)
        self.lr_frame.pack(fill='x', pady=2)

        # Override checkbox
        self.override_lr_var = tk.BooleanVar(value=False)
        self.override_lr_checkbox = ttk.Checkbutton(
            self.lr_frame,
            text="Override checkpoint LR",
            variable=self.override_lr_var,
            command=self._on_override_lr_toggle
        )
        self.override_lr_checkbox.pack(anchor='w', pady=(0, 5))

        lr_row = ttk.Frame(self.lr_frame)
        lr_row.pack(fill='x', pady=2)
        ttk.Label(lr_row, text="Learning Rate:", width=14).pack(side='left')
        self.lr_var = tk.StringVar(value="0.0002")
        self.lr_entry = ttk.Entry(lr_row, textvariable=self.lr_var, width=12, state='disabled')
        self.lr_entry.pack(side='left', padx=(0, 5))

        # Add helpful note
        lr_note = ttk.Label(
            self.lr_frame,
            text="Default: 2e-4. Use 1e-5 to 1e-4 for fine-tuning.",
            font=('TkDefaultFont', 8),
            foreground='gray'
        )
        lr_note.pack(anchor='w', padx=(0, 0))

        # Optimizer (left column, below DME settings)
        self.optimizer_frame = ttk.LabelFrame(left_col, text="Optimizer", padding=5)
        self.optimizer_frame.pack(fill='x', pady=2)

        opt_row = ttk.Frame(self.optimizer_frame)
        opt_row.pack(fill='x', pady=2)
        ttk.Label(opt_row, text="Type:", width=14).pack(side='left')
        self.optimizer_var = tk.StringVar(value="adam")
        ttk.Combobox(opt_row, textvariable=self.optimizer_var,
                     values=["adam", "adamw", "sgd"], state="readonly",
                     width=10).pack(side='left')

        beta_row = ttk.Frame(self.optimizer_frame)
        beta_row.pack(fill='x', pady=2)
        ttk.Label(beta_row, text="β1 / momentum:", width=14).pack(side='left')
        self.adam_beta1_var = tk.StringVar(value="0.9")
        ttk.Entry(beta_row, textvariable=self.adam_beta1_var, width=8).pack(side='left', padx=(0, 8))
        ttk.Label(beta_row, text="β2:", width=4).pack(side='left')
        self.adam_beta2_var = tk.StringVar(value="0.999")
        ttk.Entry(beta_row, textvariable=self.adam_beta2_var, width=8).pack(side='left')

        wd_row = ttk.Frame(self.optimizer_frame)
        wd_row.pack(fill='x', pady=2)
        ttk.Label(wd_row, text="Weight decay:", width=14).pack(side='left')
        self.weight_decay_var = tk.StringVar(value="0.0")
        ttk.Entry(wd_row, textvariable=self.weight_decay_var, width=10).pack(side='left')

        ttk.Label(self.optimizer_frame,
                  text="Use AdamW for L2 regularisation; SGD if Adam is unstable.",
                  font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', pady=(2, 0))

        # LR scheduler (right column, below LR)
        self.scheduler_frame = ttk.LabelFrame(right_col, text="LR Schedule", padding=5)
        self.scheduler_frame.pack(fill='x', pady=2)

        sched_row = ttk.Frame(self.scheduler_frame)
        sched_row.pack(fill='x', pady=2)
        ttk.Label(sched_row, text="Type:", width=14).pack(side='left')
        self.lr_schedule_var = tk.StringVar(value="step")
        ttk.Combobox(sched_row, textvariable=self.lr_schedule_var,
                     values=["none", "step", "exponential", "cosine"], state="readonly",
                     width=12).pack(side='left')

        decay_row = ttk.Frame(self.scheduler_frame)
        decay_row.pack(fill='x', pady=2)
        ttk.Label(decay_row, text="Decay start (epoch):", width=18).pack(side='left')
        self.lr_decay_start_var = tk.StringVar(value="200")
        ttk.Entry(decay_row, textvariable=self.lr_decay_start_var, width=8).pack(side='left')

        rate_row = ttk.Frame(self.scheduler_frame)
        rate_row.pack(fill='x', pady=2)
        ttk.Label(rate_row, text="Decay rate:", width=14).pack(side='left')
        self.lr_decay_rate_var = tk.StringVar(value="0.005")
        ttk.Entry(rate_row, textvariable=self.lr_decay_rate_var, width=10).pack(side='left')
        ttk.Label(rate_row, text="Min LR:", width=8).pack(side='left', padx=(8, 0))
        self.lr_min_var = tk.StringVar(value="1e-6")
        ttk.Entry(rate_row, textvariable=self.lr_min_var, width=10).pack(side='left')

        # Regularisation (left column, below optimizer)
        self.reg_frame = ttk.LabelFrame(left_col, text="Regularisation", padding=5)
        self.reg_frame.pack(fill='x', pady=2)

        clip_row = ttk.Frame(self.reg_frame)
        clip_row.pack(fill='x', pady=2)
        ttk.Label(clip_row, text="Grad clip norm:", width=14).pack(side='left')
        self.grad_clip_var = tk.StringVar(value="0.0")
        ttk.Entry(clip_row, textvariable=self.grad_clip_var, width=10).pack(side='left')
        ttk.Label(self.reg_frame,
                  text="0 = disabled. Try 1.0 if loss spikes.",
                  font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', pady=(2, 0))

        # Loss / reproducibility (left column)
        self.loss_frame = ttk.LabelFrame(left_col, text="Loss & Reproducibility", padding=5)
        self.loss_frame.pack(fill='x', pady=2)

        eps_row = ttk.Frame(self.loss_frame)
        eps_row.pack(fill='x', pady=2)
        ttk.Label(eps_row, text="Log eps:", width=14).pack(side='left')
        self.log_eps_var = tk.StringVar(value="0.01")
        ttk.Entry(eps_row, textvariable=self.log_eps_var, width=10).pack(side='left')

        seed_row = ttk.Frame(self.loss_frame)
        seed_row.pack(fill='x', pady=2)
        ttk.Label(seed_row, text="Random seed:", width=14).pack(side='left')
        self.seed_var = tk.StringVar(value="42")
        ttk.Entry(seed_row, textvariable=self.seed_var, width=10).pack(side='left')

        # GPU setting (right column)
        self.gpu_frame = ttk.LabelFrame(right_col, text="Device", padding=5)
        self.gpu_frame.pack(fill='x', pady=2)

        self.use_gpu_var = tk.BooleanVar(value=True)
        self.gpu_checkbox = ttk.Checkbutton(
            self.gpu_frame, text="Use GPU (if available)",
            variable=self.use_gpu_var
        )
        self.gpu_checkbox.pack(anchor='w')

        self.cuda_launch_blocking_var = tk.BooleanVar(value=False)
        self.cuda_blocking_checkbox = ttk.Checkbutton(
            self.gpu_frame, text="CUDA Launch Blocking (debugging)",
            variable=self.cuda_launch_blocking_var
        )
        self.cuda_blocking_checkbox.pack(anchor='w')

        # Add tooltip/explanation
        blocking_help = ttk.Label(
            self.gpu_frame,
            text="(Forces sequential execution - slower but shows exact error locations)",
            font=('TkDefaultFont', 8),
            foreground='gray'
        )
        blocking_help.pack(anchor='w', padx=(20, 0))

        # Checkpoint saving options (right column)
        self.checkpoint_frame = ttk.LabelFrame(right_col, text="Checkpoint Saving", padding=5)
        self.checkpoint_frame.pack(fill='x', pady=2)

        self.save_only_best_var = tk.BooleanVar(value=False)
        self.save_only_best_checkbox = ttk.Checkbutton(
            self.checkpoint_frame,
            text="Save only best checkpoints (skip per-epoch)",
            variable=self.save_only_best_var
        )
        self.save_only_best_checkbox.pack(anchor='w')

        # Add tooltip/explanation
        save_help = ttk.Label(
            self.checkpoint_frame,
            text="(Saves disk space: ~20 MB per DD epoch, ~160 KB per DME epoch)",
            font=('TkDefaultFont', 8),
            foreground='gray'
        )
        save_help.pack(anchor='w', padx=(20, 0))

        # Validation Split Strategy
        self.val_split_frame = ttk.LabelFrame(
            self.tab_train, text="Validation Split Strategy", padding=10)
        self.val_split_frame.pack(fill='x', pady=5)

        self.val_split_var = tk.StringVar(value="random")

        self.val_split_random_radio = ttk.Radiobutton(
            self.val_split_frame,
            text="Random (seed-based)",
            variable=self.val_split_var,
            value="random"
        )
        self.val_split_random_radio.pack(anchor='w', pady=2)

        self.val_split_stratified_radio = ttk.Radiobutton(
            self.val_split_frame,
            text="Stratified (balanced by CoC bins)",
            variable=self.val_split_var,
            value="stratified"
        )
        self.val_split_stratified_radio.pack(anchor='w', pady=2)

        # Info label
        val_split_info = ttk.Label(
            self.val_split_frame,
            text="ℹ️ Only changeable when starting fresh. Locked when resuming from checkpoint.",
            font=('TkDefaultFont', 8),
            foreground='gray'
        )
        val_split_info.pack(anchor='w', pady=(5, 0))

        # Data loading settings (right column)
        # Note: num_workers removed - doesn't work when training is launched from GUI on Windows
        # PyTorch DataLoader requires multiprocessing "spawn" which needs if __name__ == '__main__'
        # Training will use num_workers=0 (single process data loading)

        # Start button frame
        start_frame = ttk.Frame(self.tab_train)
        start_frame.pack(fill='x', pady=10)

        self.start_train_btn = ttk.Button(
            start_frame, text="Start Training",
            command=self._start_training,
            state='disabled'
        )
        self.start_train_btn.pack(side='left', padx=5)

        self.stop_btn = ttk.Button(
            start_frame, text="Stop",
            command=self._stop_training,
            state='disabled'
        )
        self.stop_btn.pack(side='left', padx=5)

        # Progress
        progress_frame = ttk.LabelFrame(self.tab_train, text="Progress", padding=10)
        progress_frame.pack(fill='x', pady=5)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=5)

        self.status_var = tk.StringVar(value="Select a training mode to begin")
        ttk.Label(progress_frame, textvariable=self.status_var).pack(anchor='w')

        # Initially disable all settings
        self._disable_all_settings()

    def _select_train_mode(self, mode: str):
        """Handle training mode selection."""
        self.train_mode_var.set(mode)

        # Reset all buttons to normal style
        for btn in [self.mode_full_btn, self.mode_dme_btn]:
            btn.state(['!pressed'])

        # Clear checkpoint selection when switching modes (forces fresh auto-detection)
        self.checkpoint_path_var.set("")
        self.checkpoint_display_var.set("Auto-detect")
        self.checkpoint_info_var.set("")

        # Disable all settings first
        self._disable_all_settings()

        # Enable relevant settings based on mode
        if mode == "full":
            self.mode_desc_var.set("Train DME model (recommended for new training)")
            self._enable_dme_settings()
            self._enable_gpu_settings()
            self.mode_full_btn.state(['pressed'])
            self._auto_detect_checkpoint('full')
            # Grey out override LR checkbox (full mode always starts fresh optimizer)
            self.override_lr_checkbox.config(state='disabled')

        elif mode == "dme":
            self.mode_desc_var.set("Train only DME subnet (defocus estimation)")
            self._enable_dme_settings()
            self._enable_gpu_settings()
            self.mode_dme_btn.state(['pressed'])
            self._auto_detect_checkpoint('dme')
            # Enable override LR checkbox (can resume from checkpoint)
            self.override_lr_checkbox.config(state='normal')

        # Enable start button
        self.start_train_btn.config(state='normal')
        self.status_var.set(f"Ready to train: {mode.upper()} mode")

    def _auto_detect_checkpoint(self, mode: str):
        """Auto-detect and display the best checkpoint for the selected mode."""
        # Clear previous info message
        self.checkpoint_info_var.set("")

        if not self.checkpoint_path_var.get():  # Only auto-detect if not already set
            output_dir = Path(self.output_dir_var.get())
            checkpoints_dir = output_dir / 'checkpoints'

            if mode == 'dme':
                # DME mode: check for dme_best.pth
                best_checkpoint = checkpoints_dir / 'dme_best.pth'
            elif mode == 'full':
                # Full mode: check for DME checkpoint to resume
                dme_checkpoint = checkpoints_dir / 'dme_best.pth'

                if dme_checkpoint.exists():
                    best_checkpoint = dme_checkpoint
                else:
                    # No checkpoint found, will start fresh
                    best_checkpoint = None
            else:
                return

            if best_checkpoint and best_checkpoint.exists():
                self.checkpoint_path_var.set(str(best_checkpoint))
                self.checkpoint_display_var.set(f"Found: {best_checkpoint.name}")
                # Load LR from checkpoint and set info message
                self._set_checkpoint_info_message(str(best_checkpoint))
            else:
                self.checkpoint_path_var.set("")
                self.checkpoint_display_var.set("Not found (will start fresh)")

    def _disable_all_settings(self):
        """Disable all settings fields."""
        for widget in [self.epochs_dme_entry, self.batch_dme_entry]:
            widget.config(state='disabled')
        self.lr_entry.config(state='disabled')
        self.override_lr_checkbox.config(state='disabled')
        self.gpu_checkbox.config(state='disabled')
        self.start_train_btn.config(state='disabled')

    def _enable_dme_settings(self):
        """Enable DME settings."""
        self.epochs_dme_entry.config(state='normal')
        self.batch_dme_entry.config(state='normal')
        self.override_lr_checkbox.config(state='normal')
        # LR entry is controlled by override checkbox

    def _enable_gpu_settings(self):
        """Enable GPU checkbox."""
        self.gpu_checkbox.config(state='normal')

    def _update_val_split_state(self):
        """Update validation split radio buttons based on whether resuming from checkpoint."""
        is_resuming = bool(self.checkpoint_path_var.get())

        if is_resuming:
            # Try to read the stratified setting from the config file
            checkpoint_path = Path(self.checkpoint_path_var.get())
            # Look for config_used.yaml in the same directory as the checkpoint
            # Checkpoints are typically in output_dir/checkpoints/, config is in output_dir/
            config_path = checkpoint_path.parent.parent / 'training_config.yaml'
            if not config_path.exists():
                config_path = checkpoint_path.parent.parent / 'optical_config.yaml'

            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)

                    # Get stratified setting from config
                    stratified = config.get('training', {}).get('stratified', False)

                    # Set the radio button to match the config value (greyed out)
                    self.val_split_var.set("stratified" if stratified else "random")
                except Exception as e:
                    self._log(f"Warning: Could not read config from {config_path}: {e}")
                    # Keep current selection if config can't be read
            else:
                self._log(f"Warning: training_config.yaml not found at {config_path}")
                # Keep current selection if config not found

            # Grey out the radio buttons
            self.val_split_random_radio.config(state='disabled')
            self.val_split_stratified_radio.config(state='disabled')
        else:
            # Enable the radio buttons for fresh training
            self.val_split_random_radio.config(state='normal')
            self.val_split_stratified_radio.config(state='normal')

    def _browse_checkpoint(self):
        """Browse for a checkpoint file."""
        output_dir = Path(self.output_dir_var.get())
        checkpoints_dir = output_dir / 'checkpoints'

        initial_dir = checkpoints_dir if checkpoints_dir.exists() else output_dir

        file_path = filedialog.askopenfilename(
            title="Select Checkpoint File",
            initialdir=initial_dir,
            filetypes=[
                ("PyTorch Checkpoint", "*.pth"),
                ("All Files", "*.*")
            ]
        )

        if file_path:
            self.checkpoint_path_var.set(file_path)
            # Show just the filename in the display
            self.checkpoint_display_var.set(Path(file_path).name)

            # Analyze checkpoint and provide context-aware info message
            self._set_checkpoint_info_message(file_path)

            self._update_val_split_state()

            # Check if we should grey out separate DME checkbox
            if self.train_mode_var.get() == 'dd':
                self._check_resume_checkpoint_type()

    def _set_checkpoint_info_message(self, checkpoint_path: str):
        """Set appropriate info message based on checkpoint type and current mode."""
        from pathlib import Path
        checkpoint_name = Path(checkpoint_path).name

        mode = self.train_mode_var.get()
        if not mode:
            self.checkpoint_info_var.set(f"ℹ️ Using: {checkpoint_name}")
            return

        try:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

            # Determine checkpoint type
            has_dme = 'dme_state_dict' in checkpoint or 'model_state_dict' in checkpoint

            # Auto-load LR from checkpoint when optimizer state will be used
            if mode == 'dme' and has_dme and 'optimizer_state_dict' in checkpoint:
                opt_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
                self.lr_var.set(f"{opt_lr:.2e}")  # Format like "2.00e-05"

            if mode == 'dme':
                if has_dme:
                    self.checkpoint_info_var.set(
                        f"ℹ️ Will resume DME training from {checkpoint_name}")
                else:
                    self.checkpoint_info_var.set(
                        f"⚠️ Checkpoint format not recognized: {checkpoint_name}")

            elif mode == 'full':
                if has_dme:
                    self.checkpoint_info_var.set(f"ℹ️ Will resume DME from {checkpoint_name}")
                else:
                    self.checkpoint_info_var.set(
                        f"⚠️ Checkpoint format not recognized: {checkpoint_name}")

        except Exception as e:
            self.checkpoint_info_var.set(f"⚠️ Could not analyze {checkpoint_name}: {str(e)}")

    def _clear_checkpoint(self):
        """Clear checkpoint selection to train from scratch."""
        self.checkpoint_path_var.set("")

        # Set mode-aware display and info message
        mode = self.train_mode_var.get()
        if mode == 'dd':
            # For DD mode, auto-detect DME checkpoint
            output_dir = Path(self.output_dir_var.get())
            checkpoints_dir = output_dir / 'checkpoints'
            dme_checkpoint = checkpoints_dir / 'dme_best.pth'

            if dme_checkpoint.exists():
                self.checkpoint_display_var.set(dme_checkpoint.name)
                self.checkpoint_info_var.set(
                    f"ℹ️ Will train DD from scratch using DME from {dme_checkpoint.name}")
            else:
                self.checkpoint_display_var.set("None")
                self.checkpoint_info_var.set(
                    "⚠️ No DME checkpoint found - DD training may not work properly")
        else:
            self.checkpoint_display_var.set("None")
            self.checkpoint_info_var.set(
                "ℹ️ Will train from scratch (will overwrite checkpoints if they already exist)")

        self._update_val_split_state()

    def _scan_checkpoint_metrics(self):
        """Scan the loaded checkpoint and display its metrics."""
        checkpoint_path = self.checkpoint_path_var.get()

        # Clear previous metrics
        self.checkpoint_metrics_var.set("")

        # Check if checkpoint is selected
        if not checkpoint_path or checkpoint_path == "":
            # Try to find auto-detected checkpoint for DD mode
            mode = self.train_mode_var.get()
            if mode == 'dd':
                output_dir = Path(self.output_dir_var.get())
                checkpoints_dir = output_dir / 'checkpoints'
                dme_checkpoint = checkpoints_dir / 'dme_best.pth'
                if dme_checkpoint.exists():
                    checkpoint_path = str(dme_checkpoint)
                else:
                    self.checkpoint_metrics_var.set("⚠️ No checkpoint selected or found")
                    return
            else:
                self.checkpoint_metrics_var.set("⚠️ No checkpoint selected")
                return

        # Check if file exists
        if not Path(checkpoint_path).exists():
            self.checkpoint_metrics_var.set(f"❌ File not found: {Path(checkpoint_path).name}")
            return

        try:
            import torch

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

            # Extract metrics
            metrics_parts = []

            # Checkpoint type
            has_dme = 'dme_state_dict' in checkpoint or 'model_state_dict' in checkpoint

            if has_dme:
                ckpt_type = "DME"
            else:
                ckpt_type = "Unknown"

            metrics_parts.append(f"Type: {ckpt_type}")

            # Epoch info
            if 'epoch' in checkpoint:
                metrics_parts.append(f"Epoch: {checkpoint['epoch']}")

            # DME metrics (MAE in pixels)
            if 'val_mae_px' in checkpoint:
                metrics_parts.append(f"DME MAE: {checkpoint['val_mae_px']:.2f}px")

            # Validation loss (less important, show last)
            if 'val_loss' in checkpoint:
                metrics_parts.append(f"Val Loss: {checkpoint['val_loss']:.4f}")

            # Combine all metrics
            if metrics_parts:
                self.checkpoint_metrics_var.set(" | ".join(metrics_parts))
            else:
                self.checkpoint_metrics_var.set("✓ Checkpoint loaded (no metrics found)")

        except Exception as e:
            self.checkpoint_metrics_var.set(f"❌ Error reading checkpoint: {str(e)}")

    def _start_training(self):
        """Start training based on selected mode."""
        mode = self.train_mode_var.get()

        if mode == "full":
            self._train_model()
        elif mode == "dme":
            self._train_dme_only()

    def _create_inference_tab_scrollable(self):
        """Create scrollable inference tab wrapper."""
        # Create canvas and scrollbar
        self.inf_canvas = tk.Canvas(self.tab_inference, highlightthickness=0)
        self.inf_scrollbar = ttk.Scrollbar(
            self.tab_inference, orient="vertical", command=self.inf_canvas.yview)
        self.inf_scrollable_frame = ttk.Frame(self.inf_canvas, padding=10)

        # Configure scrolling
        self.inf_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.inf_canvas.configure(scrollregion=self.inf_canvas.bbox("all"))
        )

        self.inf_canvas.create_window((0, 0), window=self.inf_scrollable_frame, anchor="nw")
        self.inf_canvas.configure(yscrollcommand=self.inf_scrollbar.set)

        # Pack canvas and scrollbar
        self.inf_scrollbar.pack(side="right", fill="y")
        self.inf_canvas.pack(side="left", fill="both", expand=True)

        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            self.inf_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel(event):
            self.inf_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            self.inf_canvas.unbind_all("<MouseWheel>")

        self.inf_scrollable_frame.bind("<Enter>", _bind_mousewheel)
        self.inf_scrollable_frame.bind("<Leave>", _unbind_mousewheel)

        # Now create the actual content
        self._create_inference_tab()

    def _create_inference_tab(self):
        """Create inference & analysis tab."""
        # Use scrollable frame for content
        parent = self.inf_scrollable_frame if hasattr(
            self, 'inf_scrollable_frame') else self.tab_inference

        # Model selection
        model_frame = ttk.LabelFrame(parent, text="Model Selection", padding=10)
        model_frame.pack(fill='x', pady=5)

        row1 = ttk.Frame(model_frame)
        row1.pack(fill='x', pady=2)
        ttk.Label(row1, text="Model Checkpoint:", width=18).pack(side='left')
        self.inf_model_var = tk.StringVar(value="training_output/checkpoints/dme_best.pth")
        ttk.Entry(row1, textvariable=self.inf_model_var, width=60).pack(side='left', padx=5)
        ttk.Button(row1, text="Browse", command=self._browse_inference_model).pack(side='left')
        ttk.Button(row1, text="Latest run",
                   command=lambda: self._fill_latest_run_checkpoint(target='inf')).pack(
            side='left', padx=2)
        ttk.Button(
            row1, text="Scan", command=self._scan_inference_checkpoint).pack(
            side='left', padx=5)

        # Model info display (below)
        model_info_row = ttk.Frame(model_frame)
        model_info_row.pack(fill='x', pady=(5, 0))
        self.inf_model_info_var = tk.StringVar(value="")
        ttk.Label(model_info_row, textvariable=self.inf_model_info_var, foreground='gray',
                  font=('', 8)).pack(anchor='w', padx=(120, 0))

        # Image Source (raw images or cine → preprocess → crops)
        self.inf_source_frame = ttk.LabelFrame(parent, text="Image Source", padding=10)
        self.inf_source_frame.pack(fill='x', pady=5)

        src_type_row = ttk.Frame(self.inf_source_frame)
        src_type_row.pack(fill='x', pady=(0, 5))
        self.inf_source_type_var = tk.StringVar(value="crops")
        ttk.Radiobutton(
            src_type_row, text="Pre-cropped Folder", variable=self.inf_source_type_var,
            value="crops", command=self._on_inf_source_type_change).pack(
            side='left', padx=5)
        ttk.Radiobutton(
            src_type_row, text="Image Folder", variable=self.inf_source_type_var, value="folder",
            command=self._on_inf_source_type_change).pack(
            side='left', padx=5)
        ttk.Radiobutton(
            src_type_row, text=".cine Files", variable=self.inf_source_type_var, value="cine",
            command=self._on_inf_source_type_change).pack(
            side='left', padx=5)
        try:
            try:
                from Calibration.cine_loader import check_pyphantom as _cpq
            except ImportError:
                import sys as _sys
                _calib_dir = str(Path(__file__).resolve().parent.parent / 'Calibration')
                if _calib_dir not in _sys.path:
                    _sys.path.insert(0, _calib_dir)
                from cine_loader import check_pyphantom as _cpq
            _pa, _ = _cpq()
        except Exception as e:
            logging.debug(f"pyphantom check failed: {e}")
            _pa = False
        ttk.Label(src_type_row, text=f"({'✓' if _pa else '⚠'} pyphantom)", font=(
            'TkDefaultFont', 8), foreground='green' if _pa else 'orange').pack(side='left', padx=10)

        self.inf_source_container = ttk.Frame(self.inf_source_frame)
        self.inf_source_container.pack(fill='x')

        # === Image Folder Source ===
        self.inf_folder_source_frame = ttk.LabelFrame(
            self.inf_source_container, text="Image Folder Source", padding=10)

        row_f1 = ttk.Frame(self.inf_folder_source_frame)
        row_f1.pack(fill='x', pady=2)
        ttk.Label(row_f1, text="Image Folder:", width=14).pack(side='left')
        self.inf_zstack_folder_var = tk.StringVar()
        ttk.Entry(
            row_f1, textvariable=self.inf_zstack_folder_var, width=50).pack(
            side='left', padx=5)
        ttk.Button(row_f1, text="Browse", command=self._browse_inf_zstack_folder).pack(side='left')

        row_f2 = ttk.Frame(self.inf_folder_source_frame)
        row_f2.pack(fill='x', pady=2)
        ttk.Label(row_f2, text="Positions CSV:", width=14).pack(side='left')
        self.inf_positions_file_var = tk.StringVar()
        ttk.Entry(
            row_f2, textvariable=self.inf_positions_file_var, width=50).pack(
            side='left', padx=5)
        ttk.Button(row_f2, text="Browse", command=self._browse_inf_positions_file).pack(side='left')

        self.inf_positions_csv_info_var = tk.StringVar(value="")
        ttk.Label(self.inf_folder_source_frame, textvariable=self.inf_positions_csv_info_var,
                  font=('TkDefaultFont', 8), foreground='blue').pack(anchor='w', pady=(2, 0))

        row_f3 = ttk.Frame(self.inf_folder_source_frame)
        row_f3.pack(fill='x', pady=2)
        ttk.Label(row_f3, text="Focus at position:", width=14).pack(side='left')
        self.inf_folder_focus_var = tk.StringVar(value="0")
        ttk.Entry(row_f3, textvariable=self.inf_folder_focus_var, width=6).pack(side='left', padx=2)
        ttk.Label(row_f3, text="mm", font=('TkDefaultFont', 8)).pack(side='left', padx=2)

        gen_frame = ttk.Frame(self.inf_folder_source_frame)
        gen_frame.pack(fill='x', pady=(10, 0))
        ttk.Label(gen_frame, text="Or generate positions:", font=(
            'TkDefaultFont', 8, 'bold')).pack(anchor='w')
        range_row = ttk.Frame(gen_frame)
        range_row.pack(fill='x', pady=2)
        ttk.Label(range_row, text="Z min:", width=6).pack(side='left')
        self.inf_z_min_var = tk.StringVar(value="-12")
        ttk.Entry(
            range_row, textvariable=self.inf_z_min_var, width=6).pack(
            side='left', padx=(0, 5))
        ttk.Label(range_row, text="max:", width=4).pack(side='left')
        self.inf_z_max_var = tk.StringVar(value="12")
        ttk.Entry(
            range_row, textvariable=self.inf_z_max_var, width=6).pack(
            side='left', padx=(0, 5))
        ttk.Label(range_row, text="mm", font=('TkDefaultFont', 8)).pack(side='left')

        # === .cine Folder Source ===
        self.inf_cine_source_frame = ttk.LabelFrame(
            self.inf_source_container, text=".cine Folder Source", padding=10)

        cine_row1 = ttk.Frame(self.inf_cine_source_frame)
        cine_row1.pack(fill='x', pady=2)
        ttk.Label(cine_row1, text=".cine Folder:", width=14).pack(side='left')
        self.inf_cine_folder_var = tk.StringVar()
        ttk.Entry(
            cine_row1, textvariable=self.inf_cine_folder_var, width=50).pack(
            side='left', padx=5)
        ttk.Button(cine_row1, text="Browse", command=self._browse_inf_cine_folder).pack(side='left')

        cine_row1b = ttk.Frame(self.inf_cine_source_frame)
        cine_row1b.pack(fill='x', pady=2)
        ttk.Label(cine_row1b, text="Positions CSV:", width=14).pack(side='left')
        self.inf_cine_positions_csv_var = tk.StringVar()
        ttk.Entry(
            cine_row1b, textvariable=self.inf_cine_positions_csv_var, width=50).pack(
            side='left', padx=5)
        ttk.Button(
            cine_row1b, text="Browse", command=self._browse_inf_cine_positions_csv).pack(
            side='left')
        ttk.Label(self.inf_cine_source_frame,
                  text="(optional: CSV with filename, stage_position_mm columns)",
                  font=('TkDefaultFont', 8), foreground='gray').pack(anchor='w', padx=14)

        cine_row2 = ttk.Frame(self.inf_cine_source_frame)
        cine_row2.pack(fill='x', pady=5)
        ttk.Label(cine_row2, text="Stage Range:", width=14).pack(side='left')
        ttk.Label(cine_row2, text="Start:", width=5).pack(side='left')
        self.inf_stage_start_var = tk.StringVar(value="0")
        ttk.Entry(
            cine_row2, textvariable=self.inf_stage_start_var, width=6).pack(
            side='left', padx=2)
        ttk.Label(cine_row2, text="End:", width=4).pack(side='left')
        self.inf_stage_end_var = tk.StringVar(value="12")
        ttk.Entry(cine_row2, textvariable=self.inf_stage_end_var, width=6).pack(side='left', padx=2)
        ttk.Label(cine_row2, text="mm (used if no CSV)", font=('TkDefaultFont', 8),
                  foreground='gray').pack(side='left', padx=5)

        cine_row3 = ttk.Frame(self.inf_cine_source_frame)
        cine_row3.pack(fill='x', pady=2)
        ttk.Label(cine_row3, text="Focus at stage:", width=14).pack(side='left')
        self.inf_stage_focus_var = tk.StringVar(value="6")
        ttk.Entry(
            cine_row3, textvariable=self.inf_stage_focus_var, width=6).pack(
            side='left', padx=2)
        ttk.Label(cine_row3, text="mm", font=('TkDefaultFont', 8)).pack(side='left', padx=2)

        cine_row4 = ttk.Frame(self.inf_cine_source_frame)
        cine_row4.pack(fill='x', pady=2)
        ttk.Label(cine_row4, text="Frame index:", width=14).pack(side='left')
        self.inf_cine_frame_idx_var = tk.StringVar(value="0")
        ttk.Entry(
            cine_row4, textvariable=self.inf_cine_frame_idx_var, width=6).pack(
            side='left', padx=2)
        ttk.Label(cine_row4, text="(which frame from each .cine)",
                  font=('TkDefaultFont', 8), foreground='gray').pack(side='left', padx=5)

        self.inf_cine_info_var = tk.StringVar(value="")
        ttk.Label(self.inf_cine_source_frame, textvariable=self.inf_cine_info_var,
                  font=('TkDefaultFont', 8), foreground='blue').pack(anchor='w', pady=2)

        # Sphere Processing (shown when folder/cine selected)
        self.inf_sphere_frame = ttk.LabelFrame(
            self.inf_source_frame, text="Sphere Processing", padding=5)

        sphere_row1 = ttk.Frame(self.inf_sphere_frame)
        sphere_row1.pack(fill='x', pady=2)
        self.inf_upper_contour_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sphere_row1, text="Upper contour only (avoids stage)",
                        variable=self.inf_upper_contour_var).pack(side='left')

        sphere_row2 = ttk.Frame(self.inf_sphere_frame)
        sphere_row2.pack(fill='x', pady=2)
        ttk.Label(sphere_row2, text="Sphere diameter (mm):").pack(side='left')
        self.inf_sphere_diameter_mm_var = tk.StringVar(value="10.0")
        ttk.Entry(
            sphere_row2, textvariable=self.inf_sphere_diameter_mm_var, width=6).pack(
            side='left', padx=5)
        ttk.Label(sphere_row2, text="(physical size of sphere)",
                  foreground='gray', font=('TkDefaultFont', 8)).pack(side='left')

        # Load & Preprocess button (shown when folder/cine selected)
        self.inf_preproc_btn_frame = ttk.Frame(self.inf_source_frame)
        self.inf_preproc_status_var = tk.StringVar(value="")
        ttk.Button(self.inf_preproc_btn_frame, text="Load & Preprocess Images",
                   command=self._load_and_preprocess_inf_images).pack(side='left', padx=5)
        ttk.Label(self.inf_preproc_btn_frame, textvariable=self.inf_preproc_status_var,
                  foreground='blue', font=('TkDefaultFont', 8)).pack(side='left', padx=5)

        self.inf_cine_folder_loader = None

        # Input/Output paths
        paths_frame = ttk.LabelFrame(parent, text="Paths", padding=10)
        paths_frame.pack(fill='x', pady=5)

        row2 = ttk.Frame(paths_frame)
        row2.pack(fill='x', pady=2)
        ttk.Label(row2, text="Crops Directory:", width=18).pack(side='left')
        self.inf_input_var = tk.StringVar(value="")
        ttk.Entry(row2, textvariable=self.inf_input_var, width=60).pack(side='left', padx=5)
        ttk.Button(row2, text="Browse", command=self._browse_inference_input).pack(side='left')
        ttk.Button(row2, text="Scan", command=self._scan_inference_crops).pack(side='left', padx=5)

        # Crops info display (below)
        crops_info_row = ttk.Frame(paths_frame)
        crops_info_row.pack(fill='x', pady=(5, 0))
        self.inf_crops_info_var = tk.StringVar(value="")
        ttk.Label(crops_info_row, textvariable=self.inf_crops_info_var, foreground='gray',
                  font=('', 8)).pack(anchor='w', padx=(120, 0))

        row3 = ttk.Frame(paths_frame)
        row3.pack(fill='x', pady=2)
        ttk.Label(row3, text="Output Directory:", width=18).pack(side='left')
        self.inf_output_var = tk.StringVar(value="inference_results")
        ttk.Entry(row3, textvariable=self.inf_output_var, width=60).pack(side='left', padx=5)
        ttk.Button(row3, text="Browse", command=self._browse_inference_output).pack(side='left')

        # Options
        options_frame = ttk.LabelFrame(parent, text="Inference Options", padding=10)
        options_frame.pack(fill='x', pady=5)

        # Two columns for options
        left_opts = ttk.Frame(options_frame)
        left_opts.pack(side='left', fill='both', expand=True)

        right_opts = ttk.Frame(options_frame)
        right_opts.pack(side='left', fill='both', expand=True, padx=(20, 0))

        # Left column - outputs
        output_opts = ttk.LabelFrame(left_opts, text="Outputs", padding=5)
        output_opts.pack(fill='x', pady=2)

        self.inf_save_viz_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            output_opts,
            text="Save Visualizations",
            variable=self.inf_save_viz_var
        ).pack(anchor='w')

        viz_rate_row = ttk.Frame(output_opts)
        viz_rate_row.pack(fill='x', pady=(5, 2))
        ttk.Label(viz_rate_row, text="  Viz Sample Rate:", width=18).pack(side='left')
        self.inf_viz_rate_var = tk.StringVar(value="10")
        ttk.Entry(
            viz_rate_row, textvariable=self.inf_viz_rate_var, width=8).pack(
            side='left', padx=(0, 5))
        ttk.Label(viz_rate_row, text="(1 per N crops)", font=(
            'TkDefaultFont', 8), foreground='gray').pack(side='left')

        # Right column - analysis
        analysis_opts = ttk.LabelFrame(right_opts, text="Analysis & Comparison", padding=5)
        analysis_opts.pack(fill='x', pady=2)

        self.inf_by_material_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            analysis_opts,
            text="Compare by Material",
            variable=self.inf_by_material_var
        ).pack(anchor='w')

        ttk.Label(
            analysis_opts,
            text="(Creates distributions, boxplots, statistics)",
            font=('TkDefaultFont', 8),
            foreground='gray',
            wraplength=300
        ).pack(anchor='w', padx=(20, 0))

        self.inf_coc_histogram_var = tk.BooleanVar(value=True)
        self.inf_coc_histogram_cb = ttk.Checkbutton(
            analysis_opts,
            text="CoC Distribution Histograms",
            variable=self.inf_coc_histogram_var
        )
        self.inf_coc_histogram_cb.pack(anchor='w', pady=(5, 0))

        self.inf_defocus_scatter_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            analysis_opts,
            text="Defocus Scatter Plots",
            variable=self.inf_defocus_scatter_var
        ).pack(anchor='w')

        # Cross-camera scale (direct mode)
        scale_frame = ttk.LabelFrame(right_opts, text="Cross-Camera Scale (Direct Mode)", padding=5)
        scale_frame.pack(fill='x', pady=(5, 2))

        scale_row = ttk.Frame(scale_frame)
        scale_row.pack(fill='x', pady=2)
        ttk.Label(scale_row, text="Inference camera:", width=16).pack(side='left')
        self.inf_camera_scale_var = tk.StringVar(value="")
        ttk.Entry(
            scale_row, textvariable=self.inf_camera_scale_var, width=10).pack(
            side='left', padx=2)
        ttk.Label(scale_row, text="px/mm", font=('TkDefaultFont', 8)).pack(side='left', padx=2)

        ttk.Label(
            scale_frame,
            text="Auto-detected from sphere when you Load & Preprocess.\n"
            "Required for direct mode if inference camera differs from calibration camera.",
            font=('TkDefaultFont', 7),
            foreground='gray', wraplength=280).pack(
            anchor='w', pady=(2, 0))

        # Device settings
        device_frame = ttk.LabelFrame(right_opts, text="Device", padding=5)
        device_frame.pack(fill='x', pady=(5, 2))

        self.inf_use_gpu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            device_frame,
            text="Use GPU (if available)",
            variable=self.inf_use_gpu_var
        ).pack(anchor='w')

        # Run buttons
        run_frame = ttk.Frame(parent)
        run_frame.pack(fill='x', pady=10)

        self.run_inference_btn = ttk.Button(
            run_frame,
            text="Run Inference on All Materials",
            command=self._run_inference,
            style='Accent.TButton'
        )
        self.run_inference_btn.pack(side='left', padx=5)

        self.run_validation_btn = ttk.Button(
            run_frame,
            text="Run Validation Test",
            command=self._run_validation
        )
        self.run_validation_btn.pack(side='left', padx=5)

        self.stop_inference_btn = ttk.Button(
            run_frame,
            text="Stop",
            command=self._stop_inference,
            state='disabled'
        )
        self.stop_inference_btn.pack(side='left', padx=5)

        # Progress
        inf_progress_frame = ttk.LabelFrame(parent, text="Progress", padding=10)
        inf_progress_frame.pack(fill='x', pady=5)

        self.inf_progress_var = tk.DoubleVar(value=0)
        self.inf_progress_bar = ttk.Progressbar(
            inf_progress_frame, variable=self.inf_progress_var, maximum=100)
        self.inf_progress_bar.pack(fill='x', pady=5)

        self.inf_status_var = tk.StringVar(value="Ready to run inference")
        ttk.Label(inf_progress_frame, textvariable=self.inf_status_var).pack(anchor='w')

        # Results summary
        results_frame = ttk.LabelFrame(parent, text="Results Summary", padding=10)
        results_frame.pack(fill='both', expand=True, pady=5)

        self.inf_results_text = scrolledtext.ScrolledText(
            results_frame, height=8, state='disabled', wrap='word')
        self.inf_results_text.pack(fill='both', expand=True)

    def _create_validation_tab(self):
        """Create validation & testing tab."""
        # Configure grid columns for the validation tab
        self.tab_validation.columnconfigure(0, weight=1)
        self.tab_validation.columnconfigure(1, weight=0)

        # Test mode selection
        mode_frame = ttk.LabelFrame(self.tab_validation, text="Test Mode", padding=10)
        mode_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)

        self.val_mode_var = tk.StringVar(value="dme")

        self.val_mode_dme_rb = ttk.Radiobutton(
            mode_frame,
            text="DME Only (CoC Estimation)",
            variable=self.val_mode_var,
            value="dme",
            command=self._on_val_mode_change
        )
        self.val_mode_dme_rb.pack(anchor='w', pady=2)

        # Model & Data paths
        paths_frame = ttk.LabelFrame(self.tab_validation, text="Paths", padding=10)
        paths_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)

        row1 = ttk.Frame(paths_frame)
        row1.pack(fill='x', pady=2)
        ttk.Label(row1, text="Model Checkpoint:", width=18).pack(side='left')
        self.val_model_var = tk.StringVar(value="training_output/checkpoints/dme_best.pth")
        ttk.Entry(row1, textvariable=self.val_model_var, width=50).pack(side='left', padx=5)
        ttk.Button(row1, text="Browse", command=self._browse_val_model).pack(side='left')
        ttk.Button(row1, text="Latest run", command=self._fill_latest_run_checkpoint).pack(
            side='left', padx=2)
        ttk.Button(row1, text="Scan", command=self._scan_checkpoint).pack(side='left', padx=5)

        self.val_checkpoint_info_var = tk.StringVar(value="Click 'Scan' to view checkpoint info")
        ttk.Label(
            paths_frame,
            textvariable=self.val_checkpoint_info_var,
            font=('TkDefaultFont', 8),
            foreground='gray'
        ).pack(anchor='w', pady=(0, 2))

        row2 = ttk.Frame(paths_frame)
        row2.pack(fill='x', pady=2)
        ttk.Label(row2, text="Synthetic Data:", width=18).pack(side='left')
        self.val_data_var = tk.StringVar(value="training_output/synthetic_data")
        ttk.Entry(row2, textvariable=self.val_data_var, width=50).pack(side='left', padx=5)
        ttk.Button(row2, text="Browse", command=self._browse_val_data).pack(side='left')
        ttk.Button(row2, text="Latest dataset", command=self._fill_latest_dataset).pack(
            side='left', padx=2)
        ttk.Button(row2, text="Scan", command=self._scan_validation_data).pack(side='left', padx=5)

        self.val_data_info_var = tk.StringVar(value="Click 'Scan' to count available test samples")
        ttk.Label(
            paths_frame,
            textvariable=self.val_data_info_var,
            font=('TkDefaultFont', 8),
            foreground='gray'
        ).pack(anchor='w', pady=(0, 2))

        row3 = ttk.Frame(paths_frame)
        row3.pack(fill='x', pady=2)
        ttk.Label(row3, text="Output Directory:", width=18).pack(side='left')
        self.val_output_var = tk.StringVar(value="test_results")
        ttk.Entry(row3, textvariable=self.val_output_var, width=50).pack(side='left', padx=5)
        ttk.Button(row3, text="Browse", command=self._browse_val_output).pack(side='left')

        # Blur Filtering (right side, vertically centered)
        filter_frame = ttk.LabelFrame(self.tab_validation, text="CoC Filtering", padding=10)
        filter_frame.grid(row=0, column=1, rowspan=2, sticky='n', padx=5, pady=5)
        self.val_coc_filter_frame = filter_frame

        # Master enable checkbox
        self.val_enable_coc_filter_var = tk.BooleanVar(value=False)
        self.val_enable_coc_filter_checkbox = ttk.Checkbutton(
            filter_frame,
            text="Enable CoC Filtering",
            variable=self.val_enable_coc_filter_var,
            command=self._on_coc_filter_toggle
        )
        self.val_enable_coc_filter_checkbox.pack(anchor='w', pady=(0, 5))

        # Min blur threshold
        min_blur_row = ttk.Frame(filter_frame)
        min_blur_row.pack(fill='x', pady=2)
        self.val_min_blur_label = ttk.Label(min_blur_row, text="Min CoC (px):", width=14)
        self.val_min_blur_label.pack(side='left')
        self.val_min_blur_var = tk.DoubleVar(value=0.0)
        self.val_min_blur_entry = ttk.Entry(
            min_blur_row, textvariable=self.val_min_blur_var, width=8, state='readonly')
        self.val_min_blur_entry.pack(side='left', padx=5)

        # Granular filter options
        self.val_filter_opts_label = ttk.Label(
            filter_frame, text="Exclude low CoC from:", font=('TkDefaultFont', 9))
        self.val_filter_opts_label.pack(anchor='w', pady=(8, 2))

        self.val_filter_worst_pct_var = tk.BooleanVar(value=True)
        self.val_filter_worst_pct_checkbox = ttk.Checkbutton(
            filter_frame,
            text="% error worst cases",
            variable=self.val_filter_worst_pct_var,
            state='disabled'
        )
        self.val_filter_worst_pct_checkbox.pack(anchor='w', padx=(10, 0))

        self.val_filter_metrics_var = tk.BooleanVar(value=True)
        self.val_filter_metrics_checkbox = ttk.Checkbutton(
            filter_frame,
            text="Reported metrics",
            variable=self.val_filter_metrics_var,
            state='disabled'
        )
        self.val_filter_metrics_checkbox.pack(anchor='w', padx=(10, 0))

        self.val_exclude_from_test_var = tk.BooleanVar(value=False)
        self.val_exclude_from_test_checkbox = ttk.Checkbutton(
            filter_frame,
            text="Exclude from testing",
            variable=self.val_exclude_from_test_var,
            command=self._on_exclude_from_test_toggle,
            state='disabled'
        )
        self.val_exclude_from_test_checkbox.pack(anchor='w', padx=(10, 0))

        # Info label
        ttk.Label(
            filter_frame,
            text="ℹ️ Filters samples where % error is misleading",
            font=('TkDefaultFont', 8),
            foreground='gray'
        ).pack(anchor='w', pady=(5, 0))

        # Test options
        options_frame = ttk.LabelFrame(self.tab_validation, text="Test Options", padding=10)
        options_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=5)

        # Left column
        left_opts = ttk.Frame(options_frame)
        left_opts.pack(side='left', fill='both', expand=True)

        # Right column
        right_opts = ttk.Frame(options_frame)
        right_opts.pack(side='left', fill='both', expand=True, padx=(20, 0))

        # Sample selection method
        sample_method_frame = ttk.LabelFrame(left_opts, text="Sample Selection", padding=5)
        sample_method_frame.pack(fill='x', pady=2)

        self.val_sample_method_var = tk.StringVar(value="percentage")

        ttk.Radiobutton(
            sample_method_frame,
            text="Use percentage of total",
            variable=self.val_sample_method_var,
            value="percentage",
            command=self._on_sample_method_change
        ).pack(anchor='w')

        pct_row = ttk.Frame(sample_method_frame)
        pct_row.pack(fill='x', pady=2, padx=(20, 0))
        ttk.Label(pct_row, text="Percentage:", width=12).pack(side='left')
        self.val_percentage_var = tk.StringVar(value="100")
        self.val_percentage_spinbox = ttk.Spinbox(
            pct_row,
            textvariable=self.val_percentage_var,
            from_=1,
            to=100,
            width=8,
            state='normal'
        )
        self.val_percentage_spinbox.pack(side='left', padx=(0, 5))
        ttk.Label(pct_row, text="%", font=('TkDefaultFont', 10)).pack(side='left')

        ttk.Radiobutton(
            sample_method_frame,
            text="Use specific number",
            variable=self.val_sample_method_var,
            value="number",
            command=self._on_sample_method_change
        ).pack(anchor='w', pady=(5, 0))

        num_row = ttk.Frame(sample_method_frame)
        num_row.pack(fill='x', pady=2, padx=(20, 0))
        ttk.Label(num_row, text="Number:", width=12).pack(side='left')
        self.val_num_samples_var = tk.StringVar(value="100")
        self.val_num_samples_spinbox = ttk.Spinbox(
            num_row,
            textvariable=self.val_num_samples_var,
            from_=1,
            to=10000,
            width=8,
            state='disabled'
        )
        self.val_num_samples_spinbox.pack(side='left', padx=(0, 5))
        self.val_samples_range_label = ttk.Label(
            num_row, text="(max: unknown)", font=('TkDefaultFont', 8),
            foreground='gray')
        self.val_samples_range_label.pack(side='left')

        # Batch size
        batch_row = ttk.Frame(left_opts)
        batch_row.pack(fill='x', pady=(10, 2))
        ttk.Label(batch_row, text="Batch Size:", width=12).pack(side='left')
        self.val_batch_size_var = tk.StringVar(value="8")
        batch_spinbox = ttk.Spinbox(
            batch_row,
            from_=1,
            to=32,
            textvariable=self.val_batch_size_var,
            width=8
        )
        batch_spinbox.pack(side='left', padx=(0, 5))
        ttk.Label(
            batch_row, text="(Higher = faster GPU utilization)", font=('', 8)).pack(
            side='left')

        # Num workers
        workers_row = ttk.Frame(left_opts)
        workers_row.pack(fill='x', pady=2)
        ttk.Label(workers_row, text="Data Loading Workers:", width=20).pack(side='left')
        self.val_num_workers_var = tk.StringVar(value="0")
        workers_spinbox = ttk.Spinbox(
            workers_row,
            from_=0,
            to=16,
            textvariable=self.val_num_workers_var,
            width=8
        )
        workers_spinbox.pack(side='left', padx=(0, 5))
        ttk.Label(
            workers_row, text="(0 recommended for GUI)", font=('', 8),
            foreground='gray').pack(
            side='left')

        # Visualizations
        viz_frame = ttk.LabelFrame(right_opts, text="Outputs", padding=5)
        viz_frame.pack(fill='x', pady=2)

        self.val_save_viz_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            viz_frame,
            text="Save Visualizations",
            variable=self.val_save_viz_var
        ).pack(anchor='w')

        # Visualization percentage
        viz_pct_row = ttk.Frame(viz_frame)
        viz_pct_row.pack(fill='x', pady=(2, 0), padx=(20, 0))
        ttk.Label(viz_pct_row, text="Viz samples:", width=12).pack(side='left')
        self.val_viz_percent_var = tk.StringVar(value="10")
        ttk.Spinbox(
            viz_pct_row,
            textvariable=self.val_viz_percent_var,
            from_=1,
            to=100,
            width=8
        ).pack(side='left', padx=(0, 5))
        ttk.Label(viz_pct_row, text="% of samples", font=('TkDefaultFont', 9)).pack(side='left')

        self.val_save_plots_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            viz_frame,
            text="Generate Analysis Plots",
            variable=self.val_save_plots_var
        ).pack(anchor='w', pady=(2, 0))

        # Worst-case visualizations (two rows for both error types)
        worst_cases_row = ttk.Frame(viz_frame)
        worst_cases_row.pack(fill='x', pady=(5, 0))

        # Left side - first worst case type
        left_worst = ttk.Frame(worst_cases_row)
        left_worst.pack(side='left', fill='x', expand=True)

        self.val_save_worst_var = tk.BooleanVar(value=True)
        self.val_save_worst_checkbox = ttk.Checkbutton(
            left_worst,
            text="Save Worst Cases (by diameter error px)",
            variable=self.val_save_worst_var,
            command=self._on_worst_case_toggle
        )
        self.val_save_worst_checkbox.pack(anchor='w')

        worst_num_row_1 = ttk.Frame(left_worst)
        worst_num_row_1.pack(fill='x', pady=(2, 0), padx=(20, 0))
        ttk.Label(worst_num_row_1, text="Number:", width=8).pack(side='left')
        self.val_num_worst_var = tk.StringVar(value="10")
        self.val_num_worst_spinbox = ttk.Spinbox(
            worst_num_row_1,
            textvariable=self.val_num_worst_var,
            from_=1,
            to=100,
            width=8
        )
        self.val_num_worst_spinbox.pack(side='left', padx=(0, 5))
        ttk.Label(worst_num_row_1, text="worst samples",
                  font=('TkDefaultFont', 9)).pack(side='left')

        # Right side - second worst case type
        right_worst = ttk.Frame(worst_cases_row)
        right_worst.pack(side='left', fill='x', expand=True, padx=(10, 0))

        self.val_save_worst_var_2 = tk.BooleanVar(value=True)
        self.val_save_worst_checkbox_2 = ttk.Checkbutton(
            right_worst,
            text="Save Worst Cases (by CoC error %)",
            variable=self.val_save_worst_var_2,
            command=self._on_worst_case_toggle_2
        )
        self.val_save_worst_checkbox_2.pack(anchor='w')

        worst_num_row_2 = ttk.Frame(right_worst)
        worst_num_row_2.pack(fill='x', pady=(2, 0), padx=(20, 0))
        ttk.Label(worst_num_row_2, text="Number:", width=8).pack(side='left')
        self.val_num_worst_var_2 = tk.StringVar(value="10")
        self.val_num_worst_spinbox_2 = ttk.Spinbox(
            worst_num_row_2,
            textvariable=self.val_num_worst_var_2,
            from_=1,
            to=100,
            width=8
        )
        self.val_num_worst_spinbox_2.pack(side='left', padx=(0, 5))
        ttk.Label(worst_num_row_2, text="worst samples",
                  font=('TkDefaultFont', 9)).pack(side='left')

        # Device
        device_frame = ttk.LabelFrame(right_opts, text="Device", padding=5)
        device_frame.pack(fill='x', pady=(5, 2))

        self.val_use_gpu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            device_frame,
            text="Use GPU (if available)",
            variable=self.val_use_gpu_var
        ).pack(anchor='w')

        # Run button (placed between Test Options and Progress)
        run_frame = ttk.Frame(self.tab_validation)
        run_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=5, pady=10)

        self.run_val_btn = ttk.Button(
            run_frame,
            text="Run Test",
            command=self._run_validation_test,
            style='Accent.TButton'
        )
        self.run_val_btn.pack(side='left', padx=5)

        self.stop_val_btn = ttk.Button(
            run_frame,
            text="Stop",
            command=self._stop_validation,
            state='disabled'
        )
        self.stop_val_btn.pack(side='left', padx=5)

        # Progress
        val_progress_frame = ttk.LabelFrame(self.tab_validation, text="Progress", padding=10)
        val_progress_frame.grid(row=4, column=0, columnspan=2, sticky='ew', padx=5, pady=5)

        self.val_progress_var = tk.DoubleVar(value=0)
        self.val_progress_bar = ttk.Progressbar(
            val_progress_frame, variable=self.val_progress_var, maximum=100)
        self.val_progress_bar.pack(fill='x', pady=5)

        self.val_status_var = tk.StringVar(value="Select test mode and click Run Test")
        ttk.Label(val_progress_frame, textvariable=self.val_status_var).pack(anchor='w')

        # Results summary
        val_results_frame = ttk.LabelFrame(self.tab_validation, text="Test Results", padding=10)
        val_results_frame.grid(row=5, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)

        # Configure row weight for results frame to expand
        self.tab_validation.rowconfigure(5, weight=1)

        self.val_results_text = scrolledtext.ScrolledText(
            val_results_frame, height=10, state='disabled', wrap='word')
        self.val_results_text.pack(fill='both', expand=True)

        # Set initial state for worst-case controls based on default mode (DME)
        self._on_val_mode_change()

    def _create_log_section(self, parent):
        """Create log output section."""
        # Log section removed - all output goes to terminal
        pass

    # =========================================================================
    # Path Selection
    # =========================================================================
    def _browse_sharp_crops(self):
        """Browse for sharp crops directory."""
        path = filedialog.askdirectory(title="Select Sharp Crops Directory")
        if path:
            self.sharp_crops_dir = Path(path)
            self.sharp_crops_var.set(path)
            self._log(f"Sharp crops dir set: {path}")

    def _browse_output_dir(self):
        """Browse for output directory."""
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_dir = Path(path)
            self.output_dir_var.set(path)
            self._log(f"Output dir set: {path}")
            self._refresh_datasets()

            # Auto-load min_blur_px from config for validation tab
            self._load_min_blur_from_config()

    # ── Dataset selection helpers (Tab 3) ────────────────────────────────

    def _refresh_datasets(self):
        """Re-scan training_output/datasets/ and populate the dropdown."""
        from run_paths import list_datasets, find_latest_dataset
        try:
            root = Path(self.output_dir_var.get())
        except Exception:
            return
        datasets = list_datasets(root)
        # Include legacy <root>/synthetic_data/ as a virtual entry if it exists
        legacy = root / "synthetic_data"
        legacy_listed = legacy.is_dir() and (legacy / "metadata.csv").is_file()

        values = [str(p) for p in datasets]
        if legacy_listed:
            values.append(str(legacy) + "  (legacy)")
        self.dataset_combo['values'] = values

        # Auto-select latest if nothing chosen yet
        if values and not self.dataset_path_var.get():
            self.dataset_path_var.set(values[0])
            self._on_dataset_select()
        elif self.dataset_path_var.get() in values:
            self._on_dataset_select()
        else:
            self.dataset_info_var.set("No dataset selected" if not values else "Pick a dataset")

    def _browse_dataset(self):
        """Pick a dataset folder manually."""
        path = filedialog.askdirectory(title="Select dataset folder (must contain metadata.csv + blur/)")
        if path:
            self.dataset_path_var.set(path)
            current = list(self.dataset_combo['values'])
            if path not in current:
                self.dataset_combo['values'] = current + [path]
            self._on_dataset_select()

    def _on_dataset_select(self):
        """Validate the selected dataset and update info label."""
        from run_paths import validate_dataset
        raw = self.dataset_path_var.get()
        if not raw:
            self.dataset_info_var.set("No dataset selected")
            return
        # Strip trailing legacy marker
        path = Path(raw.replace("  (legacy)", "").strip())
        ok, msg = validate_dataset(path)
        if not ok:
            self.dataset_info_var.set(f"Invalid: {msg}")
            return
        # Try to read dataset_summary.json for richer info
        summary_path = path / "dataset_summary.json"
        if summary_path.is_file():
            try:
                import json
                with open(summary_path) as f:
                    s = json.load(f)
                blur_lo, blur_hi = s.get('blur_range_px', [0, 0])
                self.dataset_info_var.set(
                    f"OK | n={s.get('n_samples', '?')}, blur=[{blur_lo:.2f}, {blur_hi:.2f}] px, "
                    f"mode={s.get('training_mode', '?')}"
                )
            except Exception:
                self.dataset_info_var.set("OK")
        else:
            self.dataset_info_var.set("OK (no dataset_summary.json — legacy dataset)")

    def _fill_latest_run_checkpoint(self, target: str = 'val'):
        """Set the checkpoint field of the validation/inference tab to the
        dme_best.pth of the most recent run."""
        from run_paths import find_latest_run
        root = Path(self.output_dir_var.get())
        latest = find_latest_run(root)
        if latest is None:
            # Fall back to legacy training_output/checkpoints/dme_best.pth
            legacy = root / "checkpoints" / "dme_best.pth"
            if legacy.is_file():
                target_var = self.inf_model_var if target == 'inf' else self.val_model_var
                target_var.set(str(legacy))
                self._log(f"Set checkpoint to legacy path: {legacy}")
                return
            messagebox.showinfo("No runs", f"No runs found under {root}/runs/.")
            return
        ckpt = latest / "checkpoints" / "dme_best.pth"
        if not ckpt.is_file():
            messagebox.showwarning(
                "No checkpoint",
                f"Latest run {latest.name} has no dme_best.pth yet.")
            return
        target_var = self.inf_model_var if target == 'inf' else self.val_model_var
        target_var.set(str(ckpt))
        self._log(f"Set checkpoint to: {ckpt}")

    def _fill_latest_dataset(self):
        """Set the validation data field to the most recent dataset."""
        from run_paths import find_latest_dataset
        root = Path(self.output_dir_var.get())
        latest = find_latest_dataset(root)
        if latest is None:
            messagebox.showinfo("No datasets", f"No datasets found under {root}/datasets/.")
            return
        self.val_data_var.set(str(latest))
        self._log(f"Set validation data to: {latest}")

    def _resolve_training_paths(self):
        """Resolve (data_dir, run_dir, config) for a training launch.

        Returns (data_dir, run_dir, config_dict) or None if anything is invalid.
        Pulls the dataset from the Tab 3 dropdown (with fallback to legacy
        <root>/synthetic_data/) and creates the timestamped run folder.
        """
        from run_paths import (datasets_root, find_latest_dataset, make_run_folder_name,
                                runs_root, validate_dataset)

        output_root = Path(self.output_dir_var.get())

        # Resolve dataset
        raw = self.dataset_path_var.get().replace("  (legacy)", "").strip()
        if raw:
            data_dir = Path(raw)
        else:
            data_dir = find_latest_dataset(output_root)
            if data_dir is None:
                messagebox.showwarning(
                    "No dataset", f"No dataset found under {output_root}/datasets/.\n"
                    "Generate one in Tab 2 first.")
                return None

        ok, msg = validate_dataset(data_dir)
        if not ok:
            messagebox.showwarning("Invalid dataset", f"{data_dir}\n{msg}")
            return None

        # Build run folder
        run_name = self.run_name_var.get().strip()
        run_dir = runs_root(output_root) / make_run_folder_name(run_name or None, default='run')
        run_dir.mkdir(parents=True, exist_ok=True)

        # Load the generation config that was used to build this dataset
        gen_cfg_path = data_dir / 'generation_config.yaml'
        if gen_cfg_path.is_file():
            with open(gen_cfg_path) as f:
                config = yaml.safe_load(f) or {}
        else:
            # Legacy datasets had training_config.yaml at the parent root
            legacy_cfg = output_root / 'training_config.yaml'
            if legacy_cfg.is_file():
                with open(legacy_cfg) as f:
                    config = yaml.safe_load(f) or {}
            else:
                messagebox.showerror(
                    "Missing config",
                    f"Could not find generation_config.yaml in {data_dir} "
                    f"or training_config.yaml in {output_root}.")
                return None

        return data_dir, run_dir, config

    def _apply_gui_training_settings(self, config: dict) -> None:
        """Layer the current GUI training-tab settings onto a config dict (in place)."""
        config.setdefault('training', {})
        cfg = config['training']
        cfg['epochs_dme'] = int(self.epochs_dme_var.get())
        cfg['batch_size'] = int(self.batch_size_var.get())
        cfg['lr'] = float(self.lr_var.get())
        cfg['override_checkpoint_lr'] = self.override_lr_var.get()
        cfg['stratified'] = (self.val_split_var.get() == "stratified")
        cfg['save_only_best'] = self.save_only_best_var.get()
        cfg['optimizer'] = self.optimizer_var.get()
        cfg['adam_beta1'] = float(self.adam_beta1_var.get())
        cfg['adam_beta2'] = float(self.adam_beta2_var.get())
        cfg['weight_decay'] = float(self.weight_decay_var.get())
        cfg['lr_schedule'] = self.lr_schedule_var.get()
        cfg['lr_decay_start_epoch'] = int(self.lr_decay_start_var.get())
        cfg['lr_decay_rate'] = float(self.lr_decay_rate_var.get())
        cfg['lr_min'] = float(self.lr_min_var.get())
        cfg['grad_clip_norm'] = float(self.grad_clip_var.get())
        cfg['log_eps'] = float(self.log_eps_var.get())
        cfg['seed'] = int(self.seed_var.get())

    def _browse_direct_calibration(self):
        """Browse and load direct calibration YAML file (USER CONSTRAINT: explicit browse only)."""
        from pathlib import Path

        file_path = filedialog.askopenfilename(
            title="Select Direct Calibration Results",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            initialdir=Path.cwd() / "calibration" / "calibration_output"
        )

        if not file_path:
            return  # User cancelled

        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)

            # Recursively search entire YAML for a key, regardless of nesting
            def find_key(d, key):
                if not isinstance(d, dict):
                    return None
                if key in d:
                    return d[key]
                for v in d.values():
                    result = find_key(v, key)
                    if result is not None:
                        return result
                return None

            # Required fields — must all be present somewhere in the YAML
            rho_direct = find_key(config, 'rho_px_per_mm')
            scale_calib_px_per_mm = find_key(config, 'scale_calib_px_per_mm')
            reference_resolution = find_key(config, 'reference_resolution')
            defocus_range = find_key(config, 'defocus_range_mm')
            sigma_0 = find_key(config, 'sigma_0')
            calibration_mode = find_key(config, 'calibration_mode')

            # Validate all required fields are present
            missing = []
            if rho_direct is None:
                missing.append('rho_px_per_mm')
            if scale_calib_px_per_mm is None:
                missing.append('scale_calib_px_per_mm')
            if reference_resolution is None:
                missing.append('reference_resolution')
            if defocus_range is None:
                missing.append('defocus_range_mm')
            if sigma_0 is None:
                missing.append('sigma_0')
            if calibration_mode is None:
                missing.append('calibration_mode')

            if missing:
                messagebox.showerror("Invalid File",
                                     f"Calibration file missing required fields:\n\n"
                                     f"{', '.join(missing)}\n\n"
                                     "These can appear at any nesting level in the YAML.")
                return

            # Validate calibration_mode is direct
            if calibration_mode != 'direct':
                messagebox.showerror(
                    "Invalid Mode",
                    f"calibration_mode is '{calibration_mode}', expected 'direct'.\n\n"
                    "This loader is for direct calibration files only.")
                return

            # Sanity validation
            if rho_direct <= 0:
                messagebox.showerror(
                    "Invalid Parameters",
                    f"Invalid rho_px_per_mm: {rho_direct}\n\nMust be greater than 0.")
                return

            if sigma_0 < 0:
                messagebox.showerror("Invalid Parameters",
                                     f"Invalid sigma_0: {sigma_0}\n\nMust be >= 0.")
                return

            # Load into GUI fields
            self.direct_calib_path_var.set(str(Path(file_path).name))
            self.rho_direct_var.set(f"{rho_direct:.6f}")
            self.sigma_0_var.set(f"{sigma_0:.4f}")

            self.calib_scale_px_per_mm_var.set(f"{scale_calib_px_per_mm:.4f}")
            self.calib_reference_resolution_var.set(str(reference_resolution))

            if isinstance(defocus_range, list) and len(defocus_range) == 2:
                self.defocus_min_var.set(str(defocus_range[0]))
                self.defocus_max_var.set(str(defocus_range[1]))

            # Store full path for logging (not displayed)
            self._loaded_calib_path = file_path

            # Log success
            self._log(f"Loaded direct calibration: {Path(file_path).name}")
            self._log(f"  mode = {calibration_mode}")
            self._log(f"  ρ_direct = {rho_direct:.6f} px/mm")
            self._log(f"  σ₀ = {sigma_0:.4f} px")
            self._log(f"  scale_calib = {scale_calib_px_per_mm:.4f} px/mm")
            self._log(f"  reference_resolution = {reference_resolution} px")
            if isinstance(defocus_range, list) and len(defocus_range) == 2:
                self._log(f"  defocus range: {defocus_range[0]} to {defocus_range[1]} mm")

            messagebox.showinfo("Success",
                                f"Loaded direct calibration parameters:\n\n"
                                f"ρ_direct = {rho_direct:.6f} px/mm\n"
                                f"σ₀ = {sigma_0:.4f} px\n"
                                f"scale_calib = {scale_calib_px_per_mm:.4f} px/mm\n"
                                f"reference_resolution = {reference_resolution} px\n"
                                f"defocus_range = {defocus_range}")

        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found:\n{file_path}")
            self._log(f"Error: Calibration file not found")
        except yaml.YAMLError as e:
            messagebox.showerror("Error", f"Invalid YAML format:\n{e}")
            self._log(f"Error: Invalid YAML in calibration file")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration file:\n{e}")
            self._log(f"Error loading direct calibration: {e}")

    def _browse_inference_model(self):
        """Browse for inference model checkpoint."""
        path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        if path:
            self.inf_model_var.set(path)
            self._log(f"Inference model set: {path}")

    # -----------------------------------------------------------------------
    # Inference Image Source — backing methods
    # -----------------------------------------------------------------------

    def _on_inf_source_type_change(self):
        """Show/hide folder vs cine source frames based on radio selection."""
        source_type = self.inf_source_type_var.get()
        if source_type == "crops":
            self.inf_folder_source_frame.pack_forget()
            self.inf_cine_source_frame.pack_forget()
            self.inf_sphere_frame.pack_forget()
            self.inf_preproc_btn_frame.pack_forget()
        elif source_type == "folder":
            self.inf_cine_source_frame.pack_forget()
            self.inf_folder_source_frame.pack(in_=self.inf_source_container, fill='x', pady=5)
            self.inf_sphere_frame.pack(in_=self.inf_source_frame, fill='x', pady=5)
            self.inf_preproc_btn_frame.pack(in_=self.inf_source_frame, fill='x', pady=5)
        else:  # cine
            self.inf_folder_source_frame.pack_forget()
            self.inf_cine_source_frame.pack(in_=self.inf_source_container, fill='x', pady=5)
            self.inf_sphere_frame.pack(in_=self.inf_source_frame, fill='x', pady=5)
            self.inf_preproc_btn_frame.pack(in_=self.inf_source_frame, fill='x', pady=5)

    def _browse_inf_zstack_folder(self):
        """Browse for image folder (z-stack of sphere images)."""
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.inf_zstack_folder_var.set(folder)

    def _browse_inf_positions_file(self):
        """Browse for positions CSV for image folder source."""
        file = filedialog.askopenfilename(
            title="Select Positions CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file:
            self.inf_positions_file_var.set(file)
            self._preview_inf_positions_csv(file)

    def _preview_inf_positions_csv(self, csv_path: str):
        """Preview contents of positions CSV."""
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            if len(df.columns) < 2:
                self.inf_positions_csv_info_var.set("CSV needs at least 2 columns")
                return
            positions = df.iloc[:, 1].astype(float).values
            n = len(positions)
            if n > 1:
                import numpy as _np
                steps = _np.diff(_np.sort(positions))
                avg_step = steps.mean()
                self.inf_positions_csv_info_var.set(
                    f"CSV: {n} positions, {positions.min():.1f} to {positions.max():.1f} mm, step ~{avg_step:.2f} mm"
                )
            else:
                self.inf_positions_csv_info_var.set(f"CSV: {n} position at {positions[0]:.1f} mm")
        except Exception as e:
            self.inf_positions_csv_info_var.set(f"Error reading CSV: {e}")

    def _browse_inf_cine_folder(self):
        """Browse for folder containing .cine files."""
        folder = filedialog.askdirectory(title="Select Folder with .cine Files")
        if folder:
            self.inf_cine_folder_var.set(folder)
            self._preview_inf_cine_folder_info()

    def _preview_inf_cine_folder_info(self):
        """Preview info about selected .cine folder."""
        folder_path = self.inf_cine_folder_var.get()
        if not folder_path:
            return
        try:
            try:
                from Calibration.cine_loader import CineFolderLoader
            except ImportError:
                import sys as _sys
                _calib_dir = str(Path(__file__).resolve().parent.parent / 'Calibration')
                if _calib_dir not in _sys.path:
                    _sys.path.insert(0, _calib_dir)
                from cine_loader import CineFolderLoader
            loader = CineFolderLoader(folder_path)
            if loader.num_files > 0:
                info = loader.get_info()
                self.inf_cine_info_var.set(
                    f"Found: {info['num_files']} .cine files, {info['image_width']}x{info['image_height']} px"
                )
                self.inf_cine_folder_loader = loader
            else:
                self.inf_cine_info_var.set("No .cine files found in folder")
                self.inf_cine_folder_loader = None
        except Exception as e:
            self.inf_cine_info_var.set(f"Error scanning folder: {e}")

    def _browse_inf_cine_positions_csv(self):
        """Browse for CSV mapping .cine filenames to stage positions."""
        file = filedialog.askopenfilename(
            title="Select Positions CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file:
            self.inf_cine_positions_csv_var.set(file)
            self._preview_inf_cine_positions_csv(file)

    def _preview_inf_cine_positions_csv(self, csv_path: str):
        """Preview contents of .cine positions CSV and auto-fill stage range."""
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            if 'filename' not in df.columns:
                self.inf_cine_info_var.set("CSV missing 'filename' column")
                return
            pos_col = None
            for col in ('stage_position_mm', 'z_position_mm', 'position'):
                if col in df.columns:
                    pos_col = col
                    break
            if pos_col is None:
                self.inf_cine_info_var.set(
                    "CSV missing position column (stage_position_mm or z_position_mm)")
                return
            positions = df[pos_col].values
            self.inf_cine_info_var.set(
                f"CSV: {len(df)} entries, positions {positions.min():.1f} to {positions.max():.1f} mm"
            )
            self.inf_stage_start_var.set(f"{positions.min():.1f}")
            self.inf_stage_end_var.set(f"{positions.max():.1f}")
        except Exception as e:
            self.inf_cine_info_var.set(f"Error reading CSV: {e}")

    def _load_and_preprocess_inf_images(self):
        """Load images from folder or .cine files, process spheres, save crops, set input path."""
        source_type = self.inf_source_type_var.get()
        output_dir = self.inf_output_var.get()
        if not output_dir:
            messagebox.showerror("Error", "Please set an output directory first")
            return

        self.inf_preproc_status_var.set("Loading...")
        self.root.update_idletasks()

        import threading
        t = threading.Thread(target=self._load_and_preprocess_inf_worker,
                             args=(source_type, output_dir), daemon=True)
        t.start()

    def _load_and_preprocess_inf_worker(self, source_type: str, output_dir: str):
        """Worker thread: load images, detect/process sphere, save crops."""
        import numpy as np
        import cv2

        try:
            try:
                from Calibration.sphere_processing import process_sphere_stack
            except ImportError:
                import sys as _sys
                _calib_dir = str(Path(__file__).resolve().parent.parent / 'Calibration')
                if _calib_dir not in _sys.path:
                    _sys.path.insert(0, _calib_dir)
                from sphere_processing import process_sphere_stack
        except ImportError as e:
            self.root.after(0, lambda: self.inf_preproc_status_var.set(f"Import error: {e}"))
            return

        images = []
        filenames = []
        positions = []

        try:
            if source_type == "folder":
                folder = self.inf_zstack_folder_var.get()
                if not folder or not Path(folder).exists():
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", "Please select a valid image folder"))
                    self.root.after(0, lambda: self.inf_preproc_status_var.set(""))
                    return

                folder = Path(folder)
                extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
                image_files = []
                for ext in extensions:
                    image_files.extend(folder.glob(f'*{ext}'))
                    image_files.extend(folder.glob(f'*{ext.upper()}'))
                image_files = sorted(set(image_files))

                if not image_files:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", "No images found in folder"))
                    self.root.after(0, lambda: self.inf_preproc_status_var.set(""))
                    return

                # Load positions from CSV if provided
                positions_file = self.inf_positions_file_var.get()
                pos_dict = {}
                if positions_file and Path(positions_file).exists():
                    try:
                        import pandas as pd
                        df = pd.read_csv(positions_file)
                        filenames_col = df.iloc[:, 0].astype(str)
                        positions_col = df.iloc[:, 1].astype(float)
                        for fn, pos in zip(filenames_col, positions_col):
                            pos_dict[fn] = pos
                            pos_dict[Path(fn).stem] = pos
                    except Exception as e:
                        self._log(f"Warning: error loading positions CSV: {e}")

                n_files = len(image_files)
                try:
                    z_min = float(self.inf_z_min_var.get())
                    z_max = float(self.inf_z_max_var.get())
                    focus_offset = float(self.inf_folder_focus_var.get())
                except ValueError:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Invalid z-range",
                        "z-min, z-max, and focus offset must be valid numbers."))
                    self.root.after(0, lambda: self.inf_preproc_status_var.set(""))
                    return
                if z_min >= z_max:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Invalid z-range",
                        "z-min must be less than z-max."))
                    self.root.after(0, lambda: self.inf_preproc_status_var.set(""))
                    return

                temp_pos = []
                for i, img_path in enumerate(image_files):
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)
                        filenames.append(img_path.name)
                        csv_pos = pos_dict.get(img_path.name, pos_dict.get(img_path.stem))
                        if csv_pos is not None:
                            temp_pos.append(csv_pos - focus_offset)
                        else:
                            z = z_min + (z_max - z_min) * i / (n_files - 1) if n_files > 1 else 0
                            temp_pos.append(z - focus_offset)
                    self.root.after(0, lambda v=(i + 1) / n_files: self.inf_preproc_status_var.set(
                        f"Loading... {int(v * 100)}%"))

                positions = temp_pos

            else:  # cine
                folder_path = self.inf_cine_folder_var.get()
                if not folder_path or not Path(folder_path).exists():
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", "Please select a valid .cine folder"))
                    self.root.after(0, lambda: self.inf_preproc_status_var.set(""))
                    return

                try:
                    from cine_loader import CineFolderLoader
                except ImportError as e:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", f"cine_loader not available: {e}"))
                    self.root.after(0, lambda: self.inf_preproc_status_var.set(""))
                    return

                loader = self.inf_cine_folder_loader
                if loader is None or str(loader.folder) != folder_path:
                    loader = CineFolderLoader(folder_path)
                    self.inf_cine_folder_loader = loader

                if loader.num_files == 0:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", "No .cine files found in folder"))
                    self.root.after(0, lambda: self.inf_preproc_status_var.set(""))
                    return

                try:
                    stage_start = float(self.inf_stage_start_var.get())
                    stage_end = float(self.inf_stage_end_var.get())
                    stage_focus = float(self.inf_stage_focus_var.get())
                    frame_idx = int(self.inf_cine_frame_idx_var.get())
                except ValueError as e:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", f"Invalid parameters: {e}"))
                    self.root.after(0, lambda: self.inf_preproc_status_var.set(""))
                    return

                self.root.after(0, lambda: self.inf_preproc_status_var.set(
                                    "Extracting .cine frames..."))

                csv_path = self.inf_cine_positions_csv_var.get()
                use_csv = csv_path and Path(csv_path).exists()
                if use_csv:
                    imgs, pos_list, fnames = loader.load_with_positions_csv(
                        csv_path=csv_path, stage_offset=stage_focus, frame_idx=frame_idx)
                else:
                    imgs, pos_list, fnames = loader.load_zstack(
                        z_start=stage_start - stage_focus,
                        z_end=stage_end - stage_focus,
                        frame_idx=frame_idx)

                if not imgs:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", "No frames extracted from .cine files"))
                    self.root.after(0, lambda: self.inf_preproc_status_var.set(""))
                    return

                sorted_idx = np.argsort(pos_list)
                images = [imgs[i] for i in sorted_idx]
                positions = [pos_list[i] for i in sorted_idx]
                filenames = [fnames[i] for i in sorted_idx]

            if not images:
                self.root.after(0, lambda: messagebox.showerror("Error", "No images loaded"))
                self.root.after(0, lambda: self.inf_preproc_status_var.set(""))
                return

            self.root.after(0, lambda: self.inf_preproc_status_var.set("Processing spheres..."))

            upper_only = self.inf_upper_contour_var.get()
            processed_images, sphere_info = process_sphere_stack(
                images, upper_only=upper_only, blacken=False, flatten=True,
                flatten_mode="inference")

            if sphere_info is not None:
                cx, cy, r = sphere_info
                self._log(f"Sphere detected: centre=({cx}, {cy}), radius={r:.1f} px")
                try:
                    d_mm = float(self.inf_sphere_diameter_mm_var.get())
                    if d_mm > 0:
                        scale_px_per_mm = (r * 2) / d_mm
                        self._log(
                            f"Inference camera scale: (2 x {r:.1f}) / {d_mm} = {scale_px_per_mm:.2f} px/mm")
                        self.root.after(
                            0, lambda s=scale_px_per_mm: self.inf_camera_scale_var.set(f"{s:.2f}"))
                except ValueError:
                    self._log("Warning: invalid sphere diameter -- cannot compute scale")
            else:
                self._log("Warning: no sphere detected during inference preprocessing")

            # Log crop sizes
            if processed_images:
                sample = processed_images[0]
                self._log(f"Crop size: {sample.shape[1]}×{sample.shape[0]} px "
                          f"(this is the native_size used in inversion)")

            # Create timestamped inference folder and save crops into it
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            inf_run_dir = Path(output_dir) / f"inference_{timestamp}"
            out_path = inf_run_dir / "preprocessed_crops"
            out_path.mkdir(parents=True, exist_ok=True)

            self.root.after(0, lambda: self.inf_preproc_status_var.set("Saving crops..."))
            n = len(processed_images)
            for i, (img, fname, pos) in enumerate(zip(processed_images, filenames, positions)):
                stem = Path(fname).stem if fname else f"frame_{i:04d}"
                out_name = f"{stem}_z{pos:+.2f}mm.png"
                if img.dtype != np.uint8:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(str(out_path / out_name), img)
                if i % 10 == 0:
                    self.root.after(0, lambda v=(i + 1) / n: self.inf_preproc_status_var.set(
                        f"Saving... {int(v * 100)}%"))

            # Point Crops Directory at the preprocessed folder and store run dir for inference
            self._inf_run_dir = str(inf_run_dir)
            self.root.after(0, lambda p=str(out_path): self.inf_input_var.set(p))
            self.root.after(0, lambda: self.inf_preproc_status_var.set(
                f"Done — {n} crops saved to {inf_run_dir.name}/preprocessed_crops/"))
            self._log(f"Inference preprocessing complete: {n} crops at "
                      f"{processed_images[0].shape[1]}×{processed_images[0].shape[0]} px "
                      f"saved to {out_path}")

        except Exception as e:
            import traceback
            msg = f"Preprocessing failed: {e}"
            self._log(f"Error: {msg}\n{traceback.format_exc()}")
            self.root.after(0, lambda: messagebox.showerror("Preprocessing Error", msg))
            self.root.after(0, lambda: self.inf_preproc_status_var.set("Error — see log"))

    def _browse_inference_input(self):
        """Browse for inference input directory (containing material subfolders)."""
        path = filedialog.askdirectory(title="Select Crops Directory (with material subfolders)")
        if path:
            self.inf_input_var.set(path)
            self._log(f"Inference input set: {path}")

    def _browse_inference_output(self):
        """Browse for inference output directory."""
        path = filedialog.askdirectory(title="Select Output Directory for Inference Results")
        if path:
            self.inf_output_var.set(path)
            self._log(f"Inference output set: {path}")

    def _scan_inference_checkpoint(self):
        """Scan inference checkpoint and display info."""
        checkpoint_path = Path(self.inf_model_var.get())

        if not checkpoint_path.exists():
            messagebox.showerror("Error", f"Checkpoint not found: {checkpoint_path}")
            self.inf_model_info_var.set("⚠ Checkpoint file not found")
            return

        try:
            import torch

            # Load checkpoint (weights_only=True to read metadata)
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=True)

            # Extract info
            epoch = checkpoint.get('epoch', 'unknown')
            checkpoint_type = []

            if 'model_state_dict' in checkpoint or 'dme_state_dict' in checkpoint:
                checkpoint_type.append("DME")

            type_str = " + ".join(checkpoint_type) if checkpoint_type else "Unknown"
            info_parts = [f"Epoch {epoch}", f"Type: {type_str}"]

            # Show max blur/coc if available (use correct label for mode)
            ckpt_max = checkpoint.get('max_blur', checkpoint.get('max_coc'))
            if ckpt_max is not None:
                mode_label = 'max_sigma' if checkpoint.get(
                    'training_mode', 'optical') =='direct' else 'max_coc'
                info_parts.append(f"{mode_label}: {ckpt_max:.2f}px")

            # Show metrics
            metrics = []
            if 'val_mae_px' in checkpoint:
                metrics.append(f"DME MAE: {checkpoint['val_mae_px']:.2f}px")
            if 'val_psnr' in checkpoint:
                metrics.append(f"PSNR: {checkpoint['val_psnr']:.2f}dB")
            if 'val_ssim' in checkpoint:
                metrics.append(f"SSIM: {checkpoint['val_ssim']:.3f}")
            if metrics:
                info_parts.append(" | ".join(metrics))

            # Detect training mode
            training_mode = checkpoint.get('training_mode', 'optical')
            info_parts.append(f"Mode: {training_mode}")

            info_text = " | ".join(info_parts)
            self.inf_model_info_var.set(info_text)
            self._log(f"Scanned checkpoint: {checkpoint_path.name} - {info_text}")

            # Store last scanned mode for use in log messages
            self._last_scanned_mode = training_mode

            # Update all blur labels to match checkpoint mode
            self._update_mode_labels(training_mode)

        except Exception as e:
            self.inf_model_info_var.set(f"⚠ Scan error: {str(e)}")
            messagebox.showerror("Scan Error", f"Failed to read checkpoint:\n{str(e)}")
            self._log(f"Checkpoint scan error: {str(e)}")

    def _scan_inference_crops(self):
        """Scan crops directory and display statistics."""
        crops_dir = Path(self.inf_input_var.get())

        if not crops_dir.exists():
            messagebox.showerror("Error", f"Directory not found: {crops_dir}")
            self.inf_crops_info_var.set("⚠ Directory not found")
            return

        try:
            # Find all material subdirectories
            material_dirs = [d for d in crops_dir.iterdir() if d.is_dir()]

            if len(material_dirs) == 0:
                # No subfolders - check for images directly in the folder
                direct_crops = list(crops_dir.glob('*_crop.png'))
                if len(direct_crops) == 0:
                    direct_crops = list(crops_dir.glob('*.png'))

                if len(direct_crops) == 0:
                    self.inf_crops_info_var.set("⚠ No images found")
                    return

                # Single folder mode
                info_text = f"1 folder (flat) | {len(direct_crops)} images"
                self.inf_crops_info_var.set(info_text)
                self._log(f"Scanned crops directory: {info_text}")
            else:
                # Count crops per material
                total_crops = 0
                material_counts = {}

                for mat_dir in material_dirs:
                    crops = list(mat_dir.glob('*_crop.png'))
                    if len(crops) == 0:
                        crops = list(mat_dir.glob('*.png'))
                    crop_count = len(crops)
                    material_counts[mat_dir.name] = crop_count
                    total_crops += crop_count

                # Format info
                info_parts = [
                    f"{len(material_dirs)} materials",
                    f"{total_crops} total images"
                ]

                # Show top 3 materials by count
                top_materials = sorted(
                    material_counts.items(),
                    key=lambda x: x[1],
                    reverse=True) [
                    : 3]
                if top_materials:
                    top_str = ", ".join([f"{mat}: {cnt}" for mat, cnt in top_materials])
                    info_parts.append(f"Top: {top_str}")

                info_text = " | ".join(info_parts)
                self.inf_crops_info_var.set(info_text)
                self._log(f"Scanned crops directory: {info_text}")

        except Exception as e:
            self.inf_crops_info_var.set(f"⚠ Scan error: {str(e)}")
            messagebox.showerror("Scan Error", f"Failed to scan directory:\n{str(e)}")
            self._log(f"Crops scan error: {str(e)}")

    def _browse_val_model(self):
        """Browse for validation model checkpoint."""
        path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        if path:
            self.val_model_var.set(path)
            self._log(f"Validation model set: {path}")

    def _browse_val_data(self):
        """Browse for validation test data directory."""
        path = filedialog.askdirectory(
            title="Select Synthetic Data Directory (with blur/, sharp/, blur_map/)")
        if path:
            self.val_data_var.set(path)
            self._log(f"Test data set: {path}")

    def _browse_val_output(self):
        """Browse for validation output directory."""
        path = filedialog.askdirectory(title="Select Output Directory for Test Results")
        if path:
            self.val_output_var.set(path)
            self._log(f"Validation output set: {path}")

    # =========================================================================
    # Training Tab UI Callbacks
    # =========================================================================
    def _on_override_lr_toggle(self):
        """Enable/disable LR override."""
        if self.override_lr_var.get():
            self.lr_entry.config(state='normal')
        else:
            self.lr_entry.config(state='disabled')

    def _load_min_blur_from_config(self):
        """Load min_blur_px from training_config.yaml and update validation tab."""
        # Get the model checkpoint path from validation tab
        model_path = Path(self.val_model_var.get())

        # Look for training_config.yaml (or old optical_config.yaml) near the model
        # Typically: training_output/checkpoints/model.pth -> training_output/training_config.yaml
        if model_path.exists():
            config_path = model_path.parent.parent / 'training_config.yaml'
            if not config_path.exists():
                config_path = model_path.parent.parent / 'optical_config.yaml'
        else:
            if self.output_dir:
                config_path = self.output_dir / 'training_config.yaml'
                if not config_path.exists():
                    config_path = self.output_dir / 'training_config.yaml'
            else:
                return

        if not config_path.exists():
            return

        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            data_section = config.get('data', {})
            min_blur_val = data_section.get('min_blur_px', data_section.get('min_coc_px', 0.0))

            # Update the validation tab min blur value
            self.val_min_blur_var.set(min_blur_val)

            # Keep state as readonly if filter is disabled (shows value but can't edit)
            if not self.val_enable_coc_filter_var.get():
                self.val_min_blur_entry.config(state='readonly')

            _bt = "blur" if getattr(self, 'training_mode_var',
                                    None) and self.training_mode_var.get() == 'direct' else "CoC"
            self._log(f"Auto-loaded min {_bt} for validation: {min_blur_val} px from {config_path}")
        except Exception as e:
            self._log(f"Warning: Could not load min_blur_px from config: {e}")

    # =========================================================================
    # Generate Tab UI Callbacks
    # =========================================================================
    def _on_min_blur_toggle(self):
        """Enable/disable min blur entry based on checkbox state."""
        if self.min_blur_enabled_var.get():
            self.min_blur_entry.config(state='normal')
        else:
            self.min_blur_entry.config(state='disabled')

    def _on_erf_validation_toggle(self):
        """Enable/disable ERF validation count entry based on checkbox state."""
        if self.erf_validation_var.get():
            self.erf_validation_count_entry.config(state='normal')
            # Reset sync tracking so it picks up current num_samples
            self._erf_count_user_modified = False
            self.erf_validation_count_var.set(self.num_samples_var.get())
        else:
            self.erf_validation_count_entry.config(state='disabled')

    def _preview_blur_levels(self):
        """Show a preview window with sample images at different blur levels."""
        try:
            import numpy as np
            from PIL import Image, ImageTk
            from synthetic_blur import BlurParams, BlurCalculator, apply_gaussian_blur

            is_direct = self.training_mode_var.get() == "direct"
            defocus_min = float(self.defocus_min_var.get())
            defocus_max = float(self.defocus_max_var.get())

            if is_direct:
                # Direct mode: σ = ρ_direct × |z| + σ₀
                rho_direct = float(self.rho_direct_var.get())
                sigma_0 = float(self.sigma_0_var.get()) if self.sigma_0_var.get() else 0.0
                max_defocus_mag = max(abs(defocus_min), abs(defocus_max))
                max_blur = rho_direct * max_defocus_mag + sigma_0
                blur_unit = "σ"
            else:
                # Optical mode: CoC from Wang formula
                focal_length = float(self.focal_length_var.get())
                f_number = float(self.f_number_var.get())
                focus_distance = float(self.focus_distance_var.get())
                pixel_size = float(self.pixel_size_var.get())
                rho = float(self.rho_var.get())

                aperture_diameter = focal_length / f_number
                inv_u0 = 1.0 / focal_length - 1.0 / focus_distance
                imaging_distance = 1.0 / inv_u0 if abs(inv_u0) > 1e-6 else focus_distance

                optical_params = BlurParams(
                    focal_length_mm=focal_length,
                    aperture_diameter_mm=aperture_diameter,
                    focus_distance_mm=focus_distance,
                    imaging_distance_mm=imaging_distance,
                    pixel_size_mm=pixel_size,
                    rho=rho
                )

                blur_calc = BlurCalculator(optical_params)
                max_blur = max(
                    abs(blur_calc.defocus_to_coc_px(defocus_min)),
                    abs(blur_calc.defocus_to_coc_px(defocus_max))
                )
                blur_unit = "CoC"

            # Generate blur levels to preview (10 intervals in bottom 50% of range)
            num_levels = 10
            max_preview_blur = max_blur * 0.50  # Bottom 50% of blur range
            coc_levels = [max_preview_blur * i / (num_levels - 1) for i in range(num_levels)]

            # Create preview window
            preview_win = tk.Toplevel(self.root)
            preview_win.title("Blur Level Preview")
            preview_win.geometry("1250x450")

            # Try to load a random sharp crop from the dataset
            import random
            sharp_img = None
            img_source = "synthetic"

            sharp_dir = self.sharp_crops_var.get()
            if sharp_dir and Path(sharp_dir).exists():
                # Find all images recursively
                sharp_path = Path(sharp_dir)
                all_images = []
                for ext in ['*.png', '*.jpg', '*.tif', '*.bmp']:
                    all_images.extend(list(sharp_path.rglob(ext)))

                if all_images:
                    # Pick a random image
                    random_img_path = random.choice(all_images)
                    try:
                        pil_sharp = Image.open(random_img_path).convert('L')
                        sharp_img = np.array(pil_sharp).astype(np.float32) / 255.0
                        img_source = random_img_path.name
                    except Exception:
                        pass

            # Fall back to synthetic if no sharp image found
            if sharp_img is None:
                img_size = 100
                droplet_diameter = 50
                sharp_img = np.ones((img_size, img_size), dtype=np.float32)
                import cv2
                cv2.circle(
                    sharp_img, (img_size // 2, img_size // 2),
                    droplet_diameter // 2, 0.0, -1)
                img_source = "synthetic"

            # Header with image source
            header = ttk.Label(
                preview_win,
                text=f"Blur Preview - Bottom 50% ({blur_unit} 0-{max_preview_blur:.1f} px of {max_blur:.1f} px max)",
                font=('TkDefaultFont', 11, 'bold'))
            header.pack(pady=(10, 2))

            source_label = ttk.Label(
                preview_win,
                text=f"Source: {img_source}",
                font=('TkDefaultFont', 9),
                foreground='gray'
            )
            source_label.pack(pady=(0, 10))

            # Scrollable canvas for images
            canvas_container = ttk.Frame(preview_win)
            canvas_container.pack(fill='both', expand=True, padx=10, pady=5)

            canvas = tk.Canvas(canvas_container)
            scrollbar = ttk.Scrollbar(canvas_container, orient='vertical', command=canvas.yview)
            scrollbar.pack(side='right', fill='y')
            canvas.pack(side='left', fill='both', expand=True)
            canvas.configure(yscrollcommand=scrollbar.set)

            grid_frame = ttk.Frame(canvas)
            canvas_window = canvas.create_window((0, 0), window=grid_frame, anchor='nw')

            def on_frame_configure(event):
                canvas.configure(scrollregion=canvas.bbox('all'))

            def on_canvas_configure(event):
                canvas.itemconfig(canvas_window, width=event.width)

            grid_frame.bind('<Configure>', on_frame_configure)
            canvas.bind('<Configure>', on_canvas_configure)

            # Bind mousewheel scrolling
            def on_mousewheel(event):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

            canvas.bind_all('<MouseWheel>', on_mousewheel)
            preview_win.bind('<Destroy>', lambda e: canvas.unbind_all('<MouseWheel>'))

            # Store PhotoImage references to prevent garbage collection
            self._preview_images = []
            self._preview_full_images = []

            for i, blur_px in enumerate(coc_levels):
                col = i % 5
                row = i // 5

                # Calculate sigma (kernel) and defocus from blur level
                if is_direct:
                    sigma = blur_px  # blur level IS sigma in direct mode
                    defocus_mm = (
                        blur_px -sigma_0) /rho_direct if rho_direct >0 and blur_px >sigma_0 else 0.0
                else:
                    sigma = blur_calc.coc_to_sigma(blur_px)
                    defocus_mm = blur_calc.blur_to_defocus(blur_px) if blur_px > 0 else 0.0

                # Apply blur
                if sigma > 0.5:
                    blurred = apply_gaussian_blur(sharp_img, sigma)
                else:
                    blurred = sharp_img.copy()

                # Convert to displayable image
                img_uint8 = (blurred * 255).astype(np.uint8)
                pil_img_full = Image.fromarray(img_uint8, mode='L')
                pil_img_thumb = pil_img_full.resize((100, 100), Image.Resampling.NEAREST)
                photo = ImageTk.PhotoImage(pil_img_thumb)
                self._preview_images.append(photo)

                # Store full-size image for click-to-enlarge
                self._preview_full_images.append((pil_img_full, blur_px, sigma, defocus_mm))

                # Create frame for this sample
                sample_frame = ttk.Frame(grid_frame)
                sample_frame.grid(row=row, column=col, padx=10, pady=5)

                # Image label with click handler
                img_label = ttk.Label(sample_frame, image=photo, cursor='hand2')
                img_label.pack()

                # Bind click to show enlarged view
                img_idx = len(self._preview_full_images) - 1
                img_label.bind('<Button-1>', lambda e,
                               idx=img_idx: self._show_enlarged_preview(idx))

                # Info label
                if is_direct:
                    info_text = f"σ: {blur_px:.2f} px\nd: {defocus_mm:.2f} mm"
                else:
                    info_text = f"CoC: {blur_px:.1f} px | σ: {sigma:.2f} px\nd: {defocus_mm:.2f} mm"
                info_label = ttk.Label(sample_frame, text=info_text,
                                       justify='center', font=('TkDefaultFont', 8))
                info_label.pack()

            # Explanation text
            _filter_term = "blur" if is_direct else "CoC"
            explanation = ttk.Label(
                preview_win,
                text=f"Showing bottom 50% of {blur_unit} range. Set min {_filter_term} filter above where blur becomes visible.",
                font=('TkDefaultFont', 9),
                foreground='gray')
            explanation.pack(pady=10)

            # Close button
            ttk.Button(preview_win, text="Close", command=preview_win.destroy).pack(pady=5)

        except Exception as e:
            messagebox.showerror("Preview Error", f"Could not generate preview: {str(e)}")

    def _show_enlarged_preview(self, idx: int):
        """Show an enlarged view of a preview image when clicked."""
        try:
            from PIL import Image, ImageTk

            pil_img, coc_px, sigma, defocus_mm = self._preview_full_images[idx]

            # Create popup window
            enlarge_win = tk.Toplevel(self.root)
            enlarge_win.title(f"CoC: {coc_px:.1f} px | σ: {sigma:.2f} px | d: {defocus_mm:.2f} mm")

            # Scale up the image (3x or to fit screen)
            orig_size = pil_img.size
            scale = 3
            new_size = (orig_size[0] * scale, orig_size[1] * scale)
            enlarged = pil_img.resize(new_size, Image.Resampling.NEAREST)

            photo = ImageTk.PhotoImage(enlarged)
            self._enlarged_photo = photo  # Keep reference

            # Display
            label = ttk.Label(enlarge_win, image=photo)
            label.pack(padx=10, pady=10)

            info = ttk.Label(
                enlarge_win,
                text=f"CoC: {coc_px:.1f} px  |  σ: {sigma:.2f} px  |  Defocus: {defocus_mm:.2f} mm  |  Size: {orig_size[0]}x{orig_size[1]} px",
                font=('TkDefaultFont', 10)
            )
            info.pack(pady=(0, 10))

            # Close on click or escape
            enlarge_win.bind('<Button-1>', lambda e: enlarge_win.destroy())
            enlarge_win.bind('<Escape>', lambda e: enlarge_win.destroy())

        except Exception as e:
            messagebox.showerror("Error", f"Could not enlarge image: {str(e)}")

    # =========================================================================
    # Validation Tab UI Callbacks
    # =========================================================================
    def _on_sample_method_change(self):
        """Toggle between percentage and number-based sampling."""
        method = self.val_sample_method_var.get()

        if method == "percentage":
            self.val_percentage_spinbox.config(state='normal')
            self.val_num_samples_spinbox.config(state='disabled')
        else:
            self.val_percentage_spinbox.config(state='disabled')
            self.val_num_samples_spinbox.config(state='normal')

    def _on_worst_case_toggle(self):
        """Toggle worst-case number spinbox based on checkbox."""
        if self.val_save_worst_var.get():
            self.val_num_worst_spinbox.config(state='normal')
        else:
            self.val_num_worst_spinbox.config(state='disabled')

    def _on_worst_case_toggle_2(self):
        """Toggle second worst-case number spinbox based on checkbox."""
        if self.val_save_worst_var_2.get():
            self.val_num_worst_spinbox_2.config(state='normal')
        else:
            self.val_num_worst_spinbox_2.config(state='disabled')

    def _on_coc_filter_toggle(self):
        """Toggle blur filter controls based on master checkbox."""
        if self.val_enable_coc_filter_var.get():
            self.val_min_blur_entry.config(state='normal')
            self.val_exclude_from_test_checkbox.config(state='normal')
            # Let _on_exclude_from_test_toggle handle the other two checkboxes
            self._on_exclude_from_test_toggle()
        else:
            self.val_min_blur_entry.config(state='readonly')  # readonly shows value but can't edit
            self.val_filter_worst_pct_checkbox.config(state='disabled')
            self.val_filter_metrics_checkbox.config(state='disabled')
            self.val_exclude_from_test_checkbox.config(state='disabled')

    def _on_exclude_from_test_toggle(self):
        """When 'Exclude from testing' is enabled, force-enable and disable the other filter options."""
        if self.val_exclude_from_test_var.get():
            # Force both options to True and disable them
            self.val_filter_worst_pct_var.set(True)
            self.val_filter_metrics_var.set(True)
            self.val_filter_worst_pct_checkbox.config(state='disabled')
            self.val_filter_metrics_checkbox.config(state='disabled')
        else:
            # Re-enable the checkboxes only if blur filtering is enabled
            if self.val_enable_coc_filter_var.get():
                self.val_filter_worst_pct_checkbox.config(state='normal')
                self.val_filter_metrics_checkbox.config(state='normal')

    def _on_val_mode_change(self):
        """Update worst-case controls label and state based on validation mode."""
        mode = self.val_mode_var.get()

        # DME only - blur error px and blur error %
        _bt = self.blur_term if hasattr(self, 'blur_term') else "CoC"
        self.val_save_worst_checkbox.config(
            text=f"Save Worst Cases (by {_bt} error px)", state='normal')
        self.val_save_worst_checkbox_2.config(
            text=f"Save Worst Cases (by {_bt} error %)", state='normal')

        if self.val_save_worst_var.get():
            self.val_num_worst_spinbox.config(state='normal')
        else:
            self.val_num_worst_spinbox.config(state='disabled')

        if self.val_save_worst_var_2.get():
            self.val_num_worst_spinbox_2.config(state='normal')
        else:
            self.val_num_worst_spinbox_2.config(state='disabled')

        # Auto-suggest appropriate checkpoint based on mode
        self._auto_suggest_checkpoint()

    def _auto_suggest_checkpoint(self):
        """Auto-populate checkpoint field with dme_best.pth if available."""
        output_dir = Path(self.output_dir_var.get())
        checkpoints_dir = output_dir / 'checkpoints'

        if not checkpoints_dir.exists():
            return

        recommended = checkpoints_dir / 'dme_best.pth'

        if recommended.exists():
            # Make path relative to current directory
            try:
                rel_path = recommended.relative_to(Path.cwd())
                self.val_model_var.set(str(rel_path))
                self.val_checkpoint_info_var.set(f"Auto-selected: {recommended.name}")
            except ValueError:
                # Can't make relative, use absolute
                self.val_model_var.set(str(recommended))
                self.val_checkpoint_info_var.set(f"Auto-selected: {recommended.name}")

    def _scan_validation_data(self):
        """Scan synthetic_data directory and count available samples."""
        data_path = Path(self.val_data_var.get())

        if not data_path.exists():
            messagebox.showerror("Error", f"Directory not found: {data_path}")
            return

        try:
            # Count pairs in synthetic_data (blur/, sharp/, coc_map/)
            blur_dir = data_path / 'blur'
            if blur_dir.exists():
                blur_files = list(blur_dir.glob('*.png'))
                count = len(blur_files)
                self.val_data_info_var.set(f"Found {count} test image pairs")
                self._log(f"Scanned synthetic_data: {count} image pairs")
            else:
                self.val_data_info_var.set("⚠ Invalid directory (no blur/ folder)")
                messagebox.showerror(
                    "Error",
                    "Invalid synthetic_data directory.\nExpected structure: blur/, sharp/, blur_map/")
                return

            # Update spinbox max value and range label
            if count > 0:
                self.val_num_samples_spinbox.config(to=count)
                self.val_samples_range_label.config(text=f"(max: {count})")
                # Set default to 20% or 100, whichever is smaller
                default_n = min(100, max(1, int(count * 0.2)))
                self.val_num_samples_var.set(str(default_n))
            else:
                self.val_data_info_var.set("⚠ No valid samples found")
                messagebox.showwarning(
                    "Warning", "No valid sample files found in the selected directory.")

        except Exception as e:
            self.val_data_info_var.set(f"⚠ Scan error: {str(e)}")
            messagebox.showerror("Scan Error", f"Failed to scan directory:\n{str(e)}")
            self._log(f"Scan error: {str(e)}")

    def _scan_checkpoint(self):
        """Scan checkpoint file and display info based on selected mode."""
        checkpoint_path = Path(self.val_model_var.get())

        if not checkpoint_path.exists():
            messagebox.showerror("Error", f"Checkpoint not found: {checkpoint_path}")
            self.val_checkpoint_info_var.set("⚠ Checkpoint file not found")
            return

        try:
            import torch

            # Load checkpoint (weights_only=True to read metadata)
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=True)

            # Extract info
            epoch = checkpoint.get('epoch', 'unknown')
            checkpoint_type = []

            if 'model_state_dict' in checkpoint or 'dme_state_dict' in checkpoint:
                checkpoint_type.append("DME")

            type_str = " + ".join(checkpoint_type) if checkpoint_type else "Unknown"

            # Get mode-specific info
            mode = self.val_mode_var.get()
            info_parts = [f"Epoch {epoch}", f"Type: {type_str}"]

            # Show max blur/coc if available (use correct label for mode)
            ckpt_max = checkpoint.get('max_blur', checkpoint.get('max_coc'))
            if ckpt_max is not None:
                mode_label = 'max_sigma' if checkpoint.get(
                    'training_mode', 'optical') =='direct' else 'max_coc'
                info_parts.append(f"{mode_label}: {ckpt_max:.2f}px")

            # Check compatibility and show relevant metrics
            warning = ""
            if 'dme_state_dict' in checkpoint or 'model_state_dict' in checkpoint:
                if 'val_mae_px' in checkpoint:
                    info_parts.append(f"DME MAE: {checkpoint['val_mae_px']:.2f}px")
            else:
                warning = " ⚠ No DME weights"

            info_text = " | ".join(info_parts) + warning
            self.val_checkpoint_info_var.set(info_text)
            self._log(f"Scanned checkpoint: {checkpoint_path.name} - {info_text}")

            # Load min_coc from this model's config
            self._load_min_blur_from_config()

            # Detect training mode and update labels
            training_mode = checkpoint.get('training_mode', 'optical')
            self._update_mode_labels(training_mode)

            # Ensure validation blur filtering frame is always enabled
            if hasattr(self, 'val_coc_filter_frame'):
                self._set_frame_state(self.val_coc_filter_frame, 'normal')

            # Show warning dialog if incompatible
            if warning:
                messagebox.showwarning(
                    "Checkpoint Warning",
                    f"The selected checkpoint may not be suitable for {mode.upper()} mode.\n\n"
                    f"Checkpoint contains: {type_str}\n"
                    f"Selected mode: {mode.upper()}\n\n"
                    f"Recommendation:\n"
                    f"- Use dme_best.pth"
                )

        except Exception as e:
            self.val_checkpoint_info_var.set(f"⚠ Scan error: {str(e)}")
            messagebox.showerror("Scan Error", f"Failed to read checkpoint:\n{str(e)}")
            self._log(f"Checkpoint scan error: {str(e)}")

    # =========================================================================
    # Scanning
    # =========================================================================
    def _scan_sharp_crops(self):
        """Load sharp crops data from preprocessing CSVs."""
        sharp_crops = self.sharp_crops_var.get()
        if not sharp_crops:
            messagebox.showwarning("Warning", "Please select a sharp crops directory first.")
            return

        self.sharp_crops_dir = Path(sharp_crops)
        if not self.sharp_crops_dir.exists():
            messagebox.showerror("Error", f"Directory not found: {sharp_crops}")
            return

        self._log(f"\nLoading from: {sharp_crops}")

        # Scan in thread to avoid UI freeze
        def scan_thread():
            camera_filter = self.camera_filter_var.get() if hasattr(self, 'camera_filter_var') else "all"
            results = self.scanner.scan_root(self.sharp_crops_dir, self._log, camera_filter)
            self.msg_queue.put(('scan_complete', results))

        threading.Thread(target=scan_thread, daemon=True).start()

    def _on_scan_complete(self, results: Dict[str, FolderStats]):
        """Handle scan completion."""
        self.folder_stats = results

        # Check cross-folder consistency
        self.all_sizes_consistent, consistency_msg = self.scanner.check_cross_folder_consistency(
                                                                                                 results)

        # Clear and populate listbox
        self.folder_listbox.delete(0, tk.END)
        total_images = 0

        for folder_name in sorted(results.keys()):
            stats = results[folder_name]
            total_images += stats.num_images

            # Show warning if sizes inconsistent within folder
            if stats.size_consistent:
                self.folder_listbox.insert(tk.END, f"{folder_name} ({stats.num_images})")
            else:
                self.folder_listbox.insert(tk.END, f"⚠ {folder_name} ({stats.num_images})")

            # Create default config for each folder
            if folder_name not in self.folder_configs:
                config = OpticalConfig(
                    folder_name=folder_name,
                    sensor_width_px=stats.image_width,
                    sensor_height_px=stats.image_height,
                )
                self.folder_configs[folder_name] = config

        # Update summary
        num_folders = len(results)
        if self.all_sizes_consistent:
            first_stats = list(results.values())[0] if results else None
            size_str = f"{first_stats.image_width}×{first_stats.image_height}" if first_stats else "N/A"
            self.summary_var.set(f"{num_folders} folders, {total_images} images\n{size_str} ✓")
        else:
            self.summary_var.set(f"{num_folders} folders, {total_images} images\n⚠ Mixed sizes")

        self._log(f"Found {num_folders} folders, {total_images} images")
        self._log(consistency_msg)

        # If sizes inconsistent, force per-folder mode
        if not self.all_sizes_consistent:
            self.config_mode_var.set("per_folder")
            self.global_radio.config(state='disabled')
        else:
            self.global_radio.config(state='normal')

        # Select first folder
        if results:
            self.folder_listbox.selection_set(0)
            self._on_folder_select(None)

    def _remove_selected_folder(self):
        """Remove the currently selected folder from the loaded set."""
        selection = self.folder_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        display_text = self.folder_listbox.get(idx)
        # Extract folder name: strip leading "⚠ " and trailing " (N)"
        folder_name = display_text.lstrip("⚠ ").rsplit(" (", 1)[0]

        # Remove from data structures
        if hasattr(self, 'folder_stats') and folder_name in self.folder_stats:
            del self.folder_stats[folder_name]
        if folder_name in self.folder_configs:
            del self.folder_configs[folder_name]

        # Remove from listbox
        self.folder_listbox.delete(idx)

        # Update summary
        if hasattr(self, 'folder_stats') and self.folder_stats:
            num_folders = len(self.folder_stats)
            total_images = sum(s.num_images for s in self.folder_stats.values())
            first_stats = list(self.folder_stats.values())[0]
            size_str = f"{first_stats.image_width}×{first_stats.image_height}"
            self.summary_var.set(f"{num_folders} folders, {total_images} images\n{size_str} ✓")
        else:
            self.summary_var.set("No folders loaded")

        # Select next item (or previous if we deleted the last one)
        count = self.folder_listbox.size()
        if count > 0:
            new_idx = min(idx, count - 1)
            self.folder_listbox.selection_set(new_idx)
            self._on_folder_select(None)

        self._log(f"Removed folder: {folder_name}")

    def _on_folder_select(self, event):
        """Handle folder selection in listbox."""
        selection = self.folder_listbox.curselection()
        if not selection:
            return

        # Extract folder name (remove warning prefix and count suffix)
        listbox_text = self.folder_listbox.get(selection[0])
        folder_name = listbox_text.lstrip("⚠ ").split(" (")[0]
        self.selected_folder = folder_name

        # Update folder stats display
        if folder_name in self.folder_stats:
            stats = self.folder_stats[folder_name]
            self.folder_stats_text.config(state='normal')
            self.folder_stats_text.delete(1.0, tk.END)

            info_lines = [
                f"Folder: {stats.folder_name}",
                f"Images: {stats.num_images}",
            ]

            if stats.camera:
                info_lines.append(f"Camera: {stats.camera}")

            if stats.size_consistent:
                info_lines.append(f"Size: {stats.image_width}×{stats.image_height} ✓")
            else:
                info_lines.append(f"⚠ MIXED SIZES:")
                for (w, h), count in stats.sizes_found.items():
                    info_lines.append(f"   {w}×{h}: {count} images")

            if stats.has_csv_metadata:
                if stats.mean_scale_px_per_mm > 0:
                    if abs(stats.max_scale_px_per_mm - stats.min_scale_px_per_mm) < 0.5:
                        info_lines.append(f"Scale: {stats.mean_scale_px_per_mm:.1f} px/mm")
                    else:
                        info_lines.append(
                            f"Scale: {stats.mean_scale_px_per_mm:.1f} px/mm  [{stats.min_scale_px_per_mm:.1f}–{stats.max_scale_px_per_mm:.1f}]")
                if stats.mean_native_blur > 0:
                    info_lines.append(
                        f"Native blur σ: {stats.mean_native_blur:.3f} px  [{stats.min_native_blur:.3f}–{stats.max_native_blur:.3f}]")
                if stats.mean_diameter_px > 0:
                    info_lines.append(
                        f"Diameter: {stats.mean_diameter_px:.1f} px  [{stats.min_diameter_px:.0f}–{stats.max_diameter_px:.0f}]")

            if stats.has_focus_metrics:
                info_lines.append(
                    f"Laplacian: {stats.mean_laplacian:.1f} (mean)  [{stats.min_laplacian:.1f}–{stats.max_laplacian:.1f}]")
                if stats.mean_tenengrad > 0:
                    info_lines.append(f"Tenengrad: {stats.mean_tenengrad:.0f} (mean)")

            self.folder_stats_text.insert(tk.END, "\n".join(info_lines))
            self.folder_stats_text.config(state='disabled')

        # Load config values
        if folder_name in self.folder_configs:
            config = self.folder_configs[folder_name]
            self.focal_length_var.set(str(config.focal_length_mm))
            self.f_number_var.set(str(config.f_number))
            self.focus_distance_var.set(str(config.focus_distance_mm))
            self.pixel_size_var.set(str(config.pixel_size_mm))
            self.defocus_min_var.set(str(config.defocus_range_min_mm))
            self.defocus_max_var.set(str(config.defocus_range_max_mm))
            self.rho_var.set(str(config.rho))

            self._update_calculated()

    def _load_from_calibration(self):
        """Load optical parameters from a calibration results YAML file."""
        file_path = filedialog.askopenfilename(
            title="Select Calibration Results YAML",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            initialdir=Path.cwd()
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)

            # Check for training_config section (from hybrid calibration export)
            if 'training_config' in config:
                tc = config['training_config']
                optics = tc.get('optics', {})
                blur = tc.get('blur', {})
            else:
                # Direct format with optical_params
                optics = config.get('optics', config.get('optical_params', {}))
                blur = config.get('blur', {})

            # Extract values
            focal_length = optics.get('focal_length_mm')
            focus_distance = optics.get('focus_distance_mm')
            pixel_size = optics.get('pixel_size_mm')

            # Handle aperture: either aperture_diameter_mm or f_number
            if 'aperture_diameter_mm' in optics and focal_length:
                f_number = focal_length / optics['aperture_diameter_mm']
            elif 'f_number' in optics:
                f_number = optics['f_number']
            else:
                f_number = None

            # Get rho from blur section or formula_rho
            rho = blur.get('rho', config.get('formula_rho'))

            # Get defocus range if available
            defocus_range = None
            if 'training_config' in config and 'data' in config['training_config']:
                defocus_range = config['training_config']['data'].get('defocus_range_mm')
            if defocus_range is None:
                defocus_range = config.get('defocus_range_mm')

            # Get calibration reference info (for cross-resolution/camera scaling)
            calib_reference_resolution = config.get('reference_resolution')
            calib_pixel_size = None
            if 'training_config' in config and 'calibration' in config['training_config']:
                calib_info = config['training_config']['calibration']
                calib_pixel_size = calib_info.get('pixel_size_mm')
                if calib_reference_resolution is None:
                    calib_reference_resolution = calib_info.get('reference_resolution')
            # Fallback to optics pixel_size if calibration section not present
            if calib_pixel_size is None:
                calib_pixel_size = pixel_size

            # Update GUI fields
            if focal_length is not None:
                self.focal_length_var.set(str(focal_length))
            if f_number is not None:
                self.f_number_var.set(str(f_number))
            if focus_distance is not None:
                self.focus_distance_var.set(str(focus_distance))
            if pixel_size is not None:
                self.pixel_size_var.set(str(pixel_size))
            if rho is not None:
                self.rho_var.set(str(round(rho, 6)))
            if defocus_range is not None and len(defocus_range) == 2:
                self.defocus_min_var.set(str(defocus_range[0]))
                self.defocus_max_var.set(str(defocus_range[1]))

            # Update calibration reference fields (read-only)
            if calib_pixel_size is not None:
                self.calib_pixel_size_var.set(str(calib_pixel_size))
            if calib_reference_resolution is not None:
                self.calib_reference_resolution_var.set(str(calib_reference_resolution))

            # Update calculated values
            self._update_calculated()

            # Log what was loaded
            self._log(f"Loaded calibration from: {Path(file_path).name}")
            self._log(f"  F={focal_length}mm, f/{f_number}, d0={focus_distance}mm")
            self._log(f"  pixel={pixel_size}mm, rho={rho:.4f}"
                      if rho else f"  pixel={pixel_size}mm")
            if defocus_range:
                self._log(f"  defocus range: {defocus_range[0]} to {defocus_range[1]} mm")
            if calib_reference_resolution:
                self._log(
                    f"  calib reference: {calib_pixel_size}mm @ {calib_reference_resolution}px")

            msg = f"Loaded optical parameters from calibration file.\n\nrho = {rho:.4f}" if rho else "Loaded optical parameters from calibration file."
            if defocus_range:
                msg += f"\nDefocus range: {defocus_range[0]} to {defocus_range[1]} mm"
            if calib_reference_resolution:
                msg += f"\nCalibration reference: {calib_reference_resolution}px"
            messagebox.showinfo("Success", msg)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration file:\n{e}")
            self._log(f"Error loading calibration: {e}")

    def _lookup_training_pixel_size(self):
        """Open browser to search for camera pixel size datasheet."""
        # Training GUI doesn't load .cine files directly, so use generic search
        query = "[enter camera name] datasheet pixel size"
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(search_url)

    def _update_calculated(self):
        """Update calculated values display."""
        try:
            focal_length = float(self.focal_length_var.get())
            f_number = float(self.f_number_var.get())
            focus_distance = float(self.focus_distance_var.get())

            aperture = focal_length / f_number

            # Lens equation
            inv_u0 = 1.0 / focal_length - 1.0 / focus_distance
            if inv_u0 > 0:
                imaging_distance = 1.0 / inv_u0
            else:
                imaging_distance = focus_distance

            # Update display
            self.calculated_text.config(state='normal')
            self.calculated_text.delete(1.0, tk.END)
            self.calculated_text.insert(tk.END,
                                        f"Aperture Diameter: {aperture:.2f} mm\n"
                                        f"Imaging Distance: {imaging_distance:.2f} mm\n"
                                        f"Magnification: {imaging_distance/focus_distance:.3f}×"
                                        )
            self.calculated_text.config(state='disabled')

        except ValueError as e:
            self.calculated_text.config(state='normal')
            self.calculated_text.delete(1.0, tk.END)
            self.calculated_text.insert(tk.END, f"Error: {e}")
            self.calculated_text.config(state='disabled')

    def _on_mode_change(self):
        """Handle mode toggle between Global and Per Subfolder."""
        mode = self.config_mode_var.get()

        if mode == "global":
            self.save_config_btn.config(text="Save Config (Global)")
            # Disable folder selection visual feedback
            self.folder_listbox.config(state='normal')
            self._log("Mode: Global - settings will apply to all folders")
        else:
            self.save_config_btn.config(text="Save Config (This Folder)")
            self.folder_listbox.config(state='normal')
            self._log("Mode: Per Subfolder - settings apply to selected folder only")

    def _set_frame_state(self, frame, state):
        """Recursively enable/disable all child widgets in a frame."""
        for child in frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass  # some widgets (frames, labels) don't support state
            if child.winfo_children():
                self._set_frame_state(child, state)

    def _update_mode_labels(self, mode: str):
        """Update all user-facing blur labels based on training mode.

        Called from _on_training_mode_change() and checkpoint scan methods so
        labels stay correct even after restarting the GUI.
        """
        blur = "Blur" if mode == "direct" else "CoC"
        sigma = "σ" if mode == "direct" else "CoC"

        # --- Tab 2 (Generate) labels ---
        if hasattr(self, 'coc_frame'):
            self.coc_frame.configure(text=f"{blur} Sampling Distribution")
        if hasattr(self, 'min_blur_frame'):
            self.min_blur_frame.configure(text=f"Minimum {blur} Filter")
        if hasattr(self, 'min_blur_check'):
            self.min_blur_check.configure(text=f"Apply minimum {blur.lower()} threshold")
        if hasattr(self, 'min_blur_label'):
            self.min_blur_label.configure(text=f"Min {blur} (px):")
        if hasattr(self, 'beta_hint_label'):
            self.beta_hint_label.configure(text=f"(α<β favors small {blur})")

        # --- Tab 3 (Training) labels ---
        if hasattr(self, 'val_split_stratified_radio'):
            self.val_split_stratified_radio.configure(text=f"Stratified (balanced by {blur} bins)")

        # --- Tab 4 (Validation) labels ---
        if hasattr(self, 'val_coc_filter_frame'):
            self.val_coc_filter_frame.configure(text=f"{blur} Filtering")
        if hasattr(self, 'val_enable_coc_filter_checkbox'):
            self.val_enable_coc_filter_checkbox.configure(text=f"Enable {blur} Filtering")
        if hasattr(self, 'val_min_blur_label'):
            self.val_min_blur_label.configure(text=f"Min {blur} (px):")
        if hasattr(self, 'val_filter_opts_label'):
            self.val_filter_opts_label.configure(text=f"Exclude low {blur} from:")
        if hasattr(self, 'val_save_worst_checkbox_2'):
            self.val_save_worst_checkbox_2.configure(text=f"Save Worst Cases (by {blur} error %)")
        if hasattr(self, 'val_mode_dme_rb'):
            self.val_mode_dme_rb.configure(text=f"DME Only ({sigma} Estimation)")

        # --- Tab 5 (Inference) labels ---
        if hasattr(self, 'inf_coc_histogram_cb'):
            self.inf_coc_histogram_cb.configure(text=f"{blur} Distribution Histograms")

    def _on_training_mode_change(self):
        """Handle training mode change between optical and direct."""
        mode = self.training_mode_var.get()

        if mode == "optical":
            # Change lens frame title back to optical mode
            self.lens_frame.configure(text="Lens Parameters (Enter Manually)")
            # Show config mode row (Apply to toggle)
            self.config_mode_row.pack(fill='x', pady=(0, 8))
            # Show separator
            self.lens_separator.pack(fill='x', pady=5)
            # Show optical parameters
            self.optical_params_frame.pack(fill='x', pady=0)
            # Hide direct parameters
            self.direct_params_frame.pack_forget()
            # Show calibration reference frame
            self.calib_ref_frame.pack(fill='x', pady=5)
            # Show rho row
            self.rho_row.pack(fill='x', pady=2)
            # Show "Load from Calibration" button
            self.load_calib_btn.pack(side='left', padx=5)
            # Show Calculate button (optical-only)
            self.calculate_btn.pack(side='left', padx=5)
            # Show calculated values (optical-only: aperture, imaging distance, magnification)
            self.calc_values_frame.pack(fill='x', pady=5)
            # Clear direct params to prevent stale data (USER CONSTRAINT: clear on mode switch)
            self.direct_calib_path_var.set("")
            self.rho_direct_var.set("")
            self.sigma_0_var.set("")
            self.calib_scale_px_per_mm_var.set("")
            # Update description
            self.mode_desc_var.set(
                "Uses optical formula (Wang et al.) to generate blur from known optical parameters")
            self._log("Training Mode: Optical Formula")

        else:  # direct mode
            # Change lens frame title to direct mode
            self.lens_frame.configure(text="Direct Calibration Parameters")
            # Hide config mode row (Apply to toggle)
            self.config_mode_row.pack_forget()
            # Hide separator
            self.lens_separator.pack_forget()
            # Hide optical parameters (USER CONSTRAINT: hidden, not just disabled)
            self.optical_params_frame.pack_forget()
            # Show direct parameters
            self.direct_params_frame.pack(fill='x', pady=0)
            # Hide calibration reference frame
            self.calib_ref_frame.pack_forget()
            # Hide rho row
            self.rho_row.pack_forget()
            # Hide "Load from Calibration" button
            self.load_calib_btn.pack_forget()
            # Hide Calculate button (optical-only)
            self.calculate_btn.pack_forget()
            # Hide calculated values (optical-only)
            self.calc_values_frame.pack_forget()
            # Update description
            self.mode_desc_var.set(
                "Uses linear blur relationship from direct calibration data (σ = ρ_direct × |z| + σ₀)")
            self._log("Training Mode: Direct Calibration (requires calibration file)")

        # Distribution, min-filter, and beta calculator apply to both modes
        # (blur = CoC in optical, σ in direct). Ensure they're visible.
        if not self.coc_frame.winfo_manager():
            self.coc_frame.pack(fill='x', pady=(10, 5))
        if not self.min_blur_frame.winfo_manager():
            self.min_blur_frame.pack(fill='x', padx=5, pady=(0, 10))
        if not self.calc_container.winfo_manager():
            self.calc_container.pack(fill='both', expand=False, pady=5)

        # Validation blur filtering applies to both modes
        if hasattr(self, 'val_coc_filter_frame'):
            self.val_coc_filter_frame.grid()

        # Update all user-facing labels
        self._update_mode_labels(mode)

    def _save_config(self):
        """Save config based on current mode (global or per-folder)."""
        mode = self.config_mode_var.get()

        if mode == "global":
            self._apply_to_all_folders()
        else:
            self._save_folder_config()

    def _save_folder_config(self):
        """Save config for selected folder."""
        if not self.selected_folder:
            messagebox.showwarning("Warning", "No folder selected.")
            return

        try:
            config = OpticalConfig(
                folder_name=self.selected_folder,
                focal_length_mm=float(self.focal_length_var.get()),
                f_number=float(self.f_number_var.get()),
                focus_distance_mm=float(self.focus_distance_var.get()),
                pixel_size_mm=float(self.pixel_size_var.get()),
                rho=float(self.rho_var.get()),
                defocus_range_min_mm=float(self.defocus_min_var.get()),
                defocus_range_max_mm=float(self.defocus_max_var.get()),
            )

            # Get sensor dimensions from folder stats
            if self.selected_folder in self.folder_stats:
                stats = self.folder_stats[self.selected_folder]
                config.sensor_width_px = stats.image_width
                config.sensor_height_px = stats.image_height

            config.update_calculated()
            self.folder_configs[self.selected_folder] = config

            self._log(f"Saved config for: {self.selected_folder}")
            messagebox.showinfo("Saved", f"Config saved for: {self.selected_folder}")

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid value: {e}")

    def _apply_to_all_folders(self):
        """Apply current config to all folders."""
        if not self.folder_stats:
            messagebox.showwarning("Warning", "No folders scanned yet.")
            return

        try:
            base_config = {
                'focal_length_mm': float(self.focal_length_var.get()),
                'f_number': float(self.f_number_var.get()),
                'focus_distance_mm': float(self.focus_distance_var.get()),
                'pixel_size_mm': float(self.pixel_size_var.get()),
                'rho': float(self.rho_var.get()),
                'defocus_range_min_mm': float(self.defocus_min_var.get()),
                'defocus_range_max_mm': float(self.defocus_max_var.get()),
                'training_mode': self.training_mode_var.get(),  # NEW
            }

            # Add mode-specific params for direct mode
            if self.training_mode_var.get() == "direct":
                # USER CONSTRAINT: Values loaded in-memory only, written here when training starts
                try:
                    rho_direct = float(self.rho_direct_var.get())
                    sigma_0 = float(self.sigma_0_var.get())

                    base_config['rho_direct'] = rho_direct
                    base_config['sigma_0'] = sigma_0

                except (ValueError, AttributeError):
                    # This shouldn't happen if validation works, but defensive
                    raise ValueError("Direct mode selected but calibration parameters not loaded")

            # Store global config for training
            self.global_config = base_config.copy()

            # OpticalConfig is only needed for optical mode (optical formula)
            if self.training_mode_var.get() != "direct":
                optical_fields = {
                    'focal_length_mm', 'f_number', 'focus_distance_mm',
                    'pixel_size_mm', 'rho', 'defocus_range_min_mm', 'defocus_range_max_mm',
                }
                optical_config_kwargs = {k: v for k,
                    v in base_config.items() if k in optical_fields}

                for folder_name, stats in self.folder_stats.items():
                    config = OpticalConfig(
                        folder_name=folder_name,
                        sensor_width_px=stats.image_width,
                        sensor_height_px=stats.image_height,
                        **optical_config_kwargs
                    )
                    config.update_calculated()
                    self.folder_configs[folder_name] = config

            self._log(f"Applied config to {len(self.folder_stats)} folders")
            messagebox.showinfo(
                "Saved", f"Global config applied to {len(self.folder_stats)} folders")

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid value: {e}")

    # =========================================================================
    # Data Generation & Training
    # =========================================================================
    def _on_blur_distribution_change(self):
        """Enable/disable beta parameters based on distribution selection."""
        if self.blur_distribution_var.get() == "weighted":
            self.beta_alpha_entry.config(state='normal')
            self.beta_beta_entry.config(state='normal')
        else:
            self.beta_alpha_entry.config(state='disabled')
            self.beta_beta_entry.config(state='disabled')

    def _calculate_bin_weights_from_beta(self) -> list:
        """Calculate bin weights from beta distribution parameters.

        Returns:
            List of 4 weights that sum to 1.0
        """
        # Check if uniform distribution is selected
        blur_distribution = self.blur_distribution_var.get()
        if blur_distribution == 'uniform':
            return [0.25, 0.25, 0.25, 0.25]

        # For weighted distribution, calculate from beta parameters
        try:
            from scipy import stats

            # Try to get beta values from GUI
            beta_alpha = float(self.beta_alpha_var.get())
            beta_beta = float(self.beta_beta_var.get())

            # Sample from beta distribution
            num_samples = 100000
            beta_samples = stats.beta.rvs(beta_alpha, beta_beta, size=num_samples)

            # Calculate distribution across 4 equal-width bins [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]
            bin_edges = [0.0, 0.25, 0.5, 0.75, 1.0]
            weights = []

            for i in range(4):
                count = np.sum((beta_samples >= bin_edges[i]) & (beta_samples < bin_edges[i + 1]))
                weight = count / num_samples
                weights.append(weight)

            # Normalize to ensure they sum to exactly 1.0
            total = sum(weights)
            weights = [w / total for w in weights]

            return weights

        except Exception:
            # Fall back to defaults if beta values not available or error
            return [0.40, 0.30, 0.20, 0.10]

    def _calculate_beta_distribution(self):
        """Calculate and display Beta distribution statistics based on current mode config."""
        try:
            import numpy as np
            from scipy import stats

            is_direct = self.training_mode_var.get() == "direct"

            # Get beta parameters
            alpha = float(self.calc_alpha_var.get())
            beta = float(self.calc_beta_var.get())

            if alpha <= 0 or beta <= 0:
                raise ValueError("Alpha and Beta must be positive")

            defocus_min = float(self.defocus_min_var.get())
            defocus_max = float(self.defocus_max_var.get())

            if is_direct:
                # Direct mode: σ = ρ_direct × |z| + σ₀
                rho_direct = float(self.rho_direct_var.get())
                sigma_0 = float(self.sigma_0_var.get()) if self.sigma_0_var.get() else 0.0
                max_defocus_mag = max(abs(defocus_min), abs(defocus_max))
                blur_max_px = rho_direct * max_defocus_mag + sigma_0
                blur_unit = "σ"
            else:
                # Optical mode: CoC from Wang formula
                from synthetic_blur import BlurCalculator, BlurParams
                focal_length = float(self.focal_length_var.get())
                f_number = float(self.f_number_var.get())
                focus_distance = float(self.focus_distance_var.get())
                pixel_size = float(self.pixel_size_var.get())

                aperture_diameter = focal_length / f_number
                inv_u0 = 1.0 / focal_length - 1.0 / focus_distance
                imaging_distance = 1.0 / inv_u0 if abs(inv_u0) > 1e-6 else focus_distance

                optical_params = BlurParams(
                    focal_length_mm=focal_length,
                    aperture_diameter_mm=aperture_diameter,
                    focus_distance_mm=focus_distance,
                    imaging_distance_mm=imaging_distance,
                    pixel_size_mm=pixel_size
                )
                blur_calc = BlurCalculator(optical_params)
                coc_at_min = abs(blur_calc.defocus_to_coc_mm(defocus_min))
                coc_at_max = abs(blur_calc.defocus_to_coc_mm(defocus_max))
                blur_max_px = max(coc_at_min, coc_at_max) / pixel_size
                blur_unit = "CoC"

            # Apply minimum blur if enabled
            if self.min_blur_enabled_var.get():
                try:
                    min_blur_threshold = float(self.min_blur_value_var.get())
                    blur_min_px = max(0.0, min_blur_threshold)
                except ValueError:
                    blur_min_px = 0.0
            else:
                blur_min_px = 0.0

            # Sample from beta distribution and map to blur range
            num_samples = 100000
            beta_samples = stats.beta.rvs(alpha, beta, size=num_samples)

            # Map [0, 1] to blur range [blur_min_px, blur_max_px]
            blur_samples = beta_samples * (blur_max_px - blur_min_px) + blur_min_px

            # Calculate statistics
            mean_blur = np.mean(blur_samples)
            median_blur = np.median(blur_samples)
            std_blur = np.std(blur_samples)

            # Calculate distribution across intervals
            num_intervals = 4
            interval_size = (blur_max_px - blur_min_px) / num_intervals
            intervals = []
            percentages = []

            for i in range(num_intervals):
                lower = blur_min_px + i * interval_size
                upper = blur_min_px + (i + 1) * interval_size
                count = np.sum((blur_samples >= lower) & (blur_samples < upper))
                pct = (count / num_samples) * 100
                intervals.append((lower, upper))
                percentages.append(pct)

            # Format output
            output = []
            output.append(f"Beta Distribution: α={alpha:.2f}, β={beta:.2f}\n")
            output.append("=" * 45)
            if blur_min_px > 0:
                output.append(
                    f"\n{blur_unit} Range:    {blur_min_px:.1f} - {blur_max_px:.1f} px (min filter applied)")
            else:
                output.append(f"\n{blur_unit} Range:    {blur_min_px:.1f} - {blur_max_px:.1f} px")
            output.append(f"Mean {blur_unit}:     {mean_blur:.2f} px")
            output.append(f"Median {blur_unit}:   {median_blur:.2f} px")
            output.append(f"Std Dev:      {std_blur:.2f} px")
            output.append(f"\n{'Distribution:'}")
            output.append("-" * 45)

            for (lower, upper), pct in zip(intervals, percentages):
                bar_length = int(pct / 2)  # Scale for display
                bar = "█" * bar_length
                output.append(f"{lower:5.1f}-{upper:5.1f} px: {pct:5.1f}% {bar}")

            # Update display
            self.beta_calc_results.config(state='normal')
            self.beta_calc_results.delete('1.0', tk.END)
            self.beta_calc_results.insert('1.0', '\n'.join(output))
            self.beta_calc_results.config(state='disabled')

            # If "Weighted" is selected, update the generation settings
            if self.blur_distribution_var.get() == "weighted":
                self.beta_alpha_var.set(f"{alpha:.3f}")
                self.beta_beta_var.set(f"{beta:.3f}")

        except Exception as e:
            self.beta_calc_results.config(state='normal')
            self.beta_calc_results.delete('1.0', tk.END)
            self.beta_calc_results.insert(
                '1.0',
                f"Error: {str(e)}\n\nPlease check:\n- Beta α and β are positive numbers\n- Mode parameters are configured in Tab 1")
            self.beta_calc_results.config(state='disabled')

    def _reverse_calculate_beta(self):
        """Find Beta parameters that produce the target distribution."""
        try:
            import numpy as np
            from scipy import stats
            from scipy.optimize import minimize

            # Get target percentages
            target_pcts = []
            total = 0.0
            for var in self.reverse_interval_vars:
                pct = float(var.get())
                if pct < 0:
                    raise ValueError("Percentages must be non-negative")
                target_pcts.append(pct)
                total += pct

            # Normalize to 100% if needed
            if abs(total - 100.0) > 0.1:
                target_pcts = [p * 100.0 / total for p in target_pcts]

            target_pcts = np.array(target_pcts) / 100.0  # Convert to fractions

            # Define objective function: minimize difference between target and actual distribution
            def objective(params):
                alpha, beta = params
                if alpha <= 0 or beta <= 0:
                    return 1e10  # Invalid parameters

                # Sample from beta distribution
                num_samples = 50000
                samples = stats.beta.rvs(alpha, beta, size=num_samples)

                # Calculate distribution across intervals
                actual_pcts = []
                for i in range(4):
                    lower = i / 4.0
                    upper = (i + 1) / 4.0
                    count = np.sum((samples >= lower) & (samples < upper))
                    actual_pcts.append(count / num_samples)

                actual_pcts = np.array(actual_pcts)

                # Mean squared error
                return np.sum((actual_pcts - target_pcts) ** 2)

            # Initial guess: try multiple starting points
            best_result = None
            best_error = float('inf')

            for alpha_init, beta_init in [(1, 1), (2, 2), (2, 5), (5, 2), (3, 3)]:
                result = minimize(objective, [alpha_init, beta_init], method='Nelder-Mead',
                                  options={'maxiter': 500, 'xatol': 0.001})
                if result.fun < best_error and result.x[0] > 0 and result.x[1] > 0:
                    best_error = result.fun
                    best_result = result

            if best_result is None:
                raise ValueError("Could not find valid Beta parameters")

            alpha_fit, beta_fit = best_result.x

            # Verify the fit
            num_samples = 100000
            samples = stats.beta.rvs(alpha_fit, beta_fit, size=num_samples)
            actual_pcts = []
            for i in range(4):
                lower = i / 4.0
                upper = (i + 1) / 4.0
                count = np.sum((samples >= lower) & (samples < upper))
                actual_pcts.append((count / num_samples) * 100)

            # Format output
            output = []
            output.append(f"Found Beta Parameters:\n")
            output.append("=" * 40)
            output.append(f"\nα = {alpha_fit:.3f}")
            output.append(f"β = {beta_fit:.3f}")
            output.append(f"\n\nActual vs Target:")
            output.append("-" * 40)

            for i, (target, actual) in enumerate(zip(target_pcts * 100, actual_pcts)):
                diff = actual - target
                output.append(
                    f"Interval {i+1}: {actual:.1f}% (target: {target:.1f}%, diff: {diff:+.1f}%)")

            # Update display
            self.reverse_calc_results.config(state='normal')
            self.reverse_calc_results.delete('1.0', tk.END)
            self.reverse_calc_results.insert('1.0', '\n'.join(output))
            self.reverse_calc_results.config(state='disabled')

            # Update forward calculator with found values
            self.calc_alpha_var.set(f"{alpha_fit:.3f}")
            self.calc_beta_var.set(f"{beta_fit:.3f}")

            # If "Weighted" is selected, update the generation settings
            if self.blur_distribution_var.get() == "weighted":
                self.beta_alpha_var.set(f"{alpha_fit:.3f}")
                self.beta_beta_var.set(f"{beta_fit:.3f}")

        except Exception as e:
            self.reverse_calc_results.config(state='normal')
            self.reverse_calc_results.delete('1.0', tk.END)
            self.reverse_calc_results.insert(
                '1.0', f"Error: {str(e)}\n\nPlease check:\n- All percentages are non-negative numbers")
            self.reverse_calc_results.config(state='disabled')

    def _generate_data(self):
        """Generate synthetic training data."""
        if not self._validate_paths():
            return

        if not self.folder_configs:
            messagebox.showwarning(
                "Warning", "No optical configs saved. Please scan and configure first.")
            return

        self._set_training_state(True)
        self._log("\n" + "=" * 50)
        self._log("Starting synthetic data generation...")

        def generate_thread():
            try:
                self._run_generation()
                self.msg_queue.put(('generation_complete', None))
            except Exception as e:
                self.msg_queue.put(('error', str(e)))

        self.training_thread = threading.Thread(target=generate_thread, daemon=True)
        self.training_thread.start()

    def _run_generation(self):
        """Run data generation (in thread)."""
        from run_paths import datasets_root, make_run_folder_name

        output_root = Path(self.output_dir_var.get())
        output_root.mkdir(parents=True, exist_ok=True)

        # Build the dataset folder: <root>/datasets/<timestamp>_<name>/
        dataset_name = self.dataset_name_var.get().strip()
        ds_folder_name = make_run_folder_name(dataset_name or None, default='dataset')
        data_dir = datasets_root(output_root) / ds_folder_name
        data_dir.mkdir(parents=True, exist_ok=True)
        # output_dir kept as the dataset folder so existing code that writes config there works
        output_dir = data_dir

        num_samples = int(self.num_samples_var.get())
        sharp_crops_dir = Path(self.sharp_crops_var.get()) if self.sharp_crops_var.get() else None

        # Use first config for sensor dimensions only
        first_folder = list(self.folder_configs.keys())[0]
        config = self.folder_configs[first_folder]

        # Build config dict from current GUI values (not saved config)
        gui_training_mode = self.training_mode_var.get()
        gui_pixel_size = float(self.pixel_size_var.get())
        gui_defocus_min = float(self.defocus_min_var.get())
        gui_defocus_max = float(self.defocus_max_var.get())

        if gui_training_mode == "direct":
            # Direct mode: use calibration params, no optical formula needed
            gui_rho_direct = float(self.rho_direct_var.get())
            gui_sigma_0 = float(self.sigma_0_var.get())
            # Optical params not needed — set defaults to avoid errors in config saving
            gui_focal_length = 0.0
            gui_f_number = 1.0
            gui_focus_distance = 0.0
            gui_rho = 0.0
            gui_aperture_diameter = 0.0
            gui_imaging_distance = 0.0
        else:
            # Optical mode: use optical formula params
            gui_focal_length = float(self.focal_length_var.get())
            gui_f_number = float(self.f_number_var.get())
            gui_focus_distance = float(self.focus_distance_var.get())
            gui_rho = float(self.rho_var.get())
            gui_rho_direct = None
            gui_sigma_0 = None
            gui_aperture_diameter = gui_focal_length / gui_f_number
            inv_u0 = 1.0 / gui_focal_length - 1.0 / gui_focus_distance
            gui_imaging_distance = 1.0 / inv_u0 if inv_u0 > 0 else gui_focus_distance

        train_size = int(self.train_size_var.get())

        # Get training crop size (original size before resize to model size).
        # In direct mode, use the actual detected image resolution from the scanned folder.
        # In optical mode, use the manually entered training crop size field.
        if gui_training_mode == "direct":
            # Use detected image size from the first scanned folder
            try:
                first_folder = list(self.folder_stats.values())[0]
                training_crop_size = max(first_folder.image_width, first_folder.image_height)
                self._log(f"Direct mode: using detected crop size {training_crop_size}px")
            except (IndexError, AttributeError):
                training_crop_size = 960  # Sensible default for direct mode
                self._log(f"Direct mode: no folder stats, defaulting to {training_crop_size}px")
        else:
            try:
                training_crop_size = int(self.training_crop_size_var.get())
            except (ValueError, AttributeError):
                training_crop_size = 299  # Default for optical mode

        # Get calibration reference parameters (for cross-resolution/camera scaling)
        calib_pixel_size = None
        calib_reference_resolution = None
        calib_scale_px_per_mm = None
        try:
            calib_px_str = self.calib_pixel_size_var.get()
            if calib_px_str:
                calib_pixel_size = float(calib_px_str)
            calib_res_str = self.calib_reference_resolution_var.get()
            if calib_res_str:
                calib_reference_resolution = int(calib_res_str)
            calib_scale_str = self.calib_scale_px_per_mm_var.get()
            if calib_scale_str:
                calib_scale_px_per_mm = float(calib_scale_str)
        except (ValueError, AttributeError):
            pass

        # Save config using current GUI values
        config_path = output_dir / 'generation_config.yaml'
        config_dict = {
            'optics': {
                'focal_length_mm': gui_focal_length,
                'aperture_diameter_mm': gui_aperture_diameter,
                'focus_distance_mm': gui_focus_distance,
                'imaging_distance_mm': gui_imaging_distance,
                'pixel_size_mm': gui_pixel_size,
                'sensor_width_px': config.sensor_width_px,
                'sensor_height_px': config.sensor_height_px,
            },
            'blur': {
                'kernel_radius_factor': 4.0,
                'rho': gui_rho,
            },
            'data': {
                'defocus_range_mm': [gui_defocus_min, gui_defocus_max],
                'droplet_diameter_range_px': [10, 150],
                'num_samples': num_samples,
                'image_size_px': train_size,
            },
            'training': {
                'batch_size': 50,
                'epochs_dme': 400,
                'learning_rate': 0.0002,
                'pixel_size_mm': gui_pixel_size,
                'crop_size_px': training_crop_size,
                'training_mode': gui_training_mode,
            }
        }

        # Add direct mode params to config
        if gui_training_mode == "direct":
            config_dict['training']['rho_direct'] = gui_rho_direct
            config_dict['training']['sigma_0'] = gui_sigma_0
            if calib_scale_px_per_mm is not None:
                config_dict['training']['scale_calib_px_per_mm'] = calib_scale_px_per_mm
            if calib_reference_resolution is not None:
                config_dict['training']['calib_reference_resolution'] = calib_reference_resolution

        # Add calibration reference info (for cross-resolution/camera scaling)
        if calib_pixel_size is not None or calib_reference_resolution is not None:
            config_dict['calibration'] = {}
            if calib_pixel_size is not None:
                config_dict['calibration']['pixel_size_mm'] = calib_pixel_size
            if calib_reference_resolution is not None:
                config_dict['calibration']['reference_resolution'] = calib_reference_resolution

        # Add blur distribution type and beta parameters
        blur_distribution = self.blur_distribution_var.get()
        config_dict['data']['blur_distribution'] = blur_distribution

        # Only save beta parameters if weighted distribution is used
        if blur_distribution == "weighted":
            config_dict['data']['beta_alpha'] = float(self.beta_alpha_var.get())
            config_dict['data']['beta_beta'] = float(self.beta_beta_var.get())

        # Save minimum blur filter if enabled
        if self.min_blur_enabled_var.get():
            try:
                min_blur_value = float(self.min_blur_value_var.get())
                if min_blur_value > 0:
                    config_dict['data']['min_blur_px'] = min_blur_value
            except ValueError:
                pass  # Don't save if invalid

        config_dict['generation'] = {
            'save_blur_trace_metadata': self.save_blur_trace_var.get(),
            'erf_validation': self.erf_validation_var.get(),
        }

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        config_label = "direct calibration" if gui_training_mode == "direct" else "optical"
        self.msg_queue.put(('log', f"Saved {config_label} config: {config_path}"))

        # Log sharp crops info
        if sharp_crops_dir and sharp_crops_dir.exists():
            # Count images in sharp crops directory (efficient single pass)
            image_count = sum(1 for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
                              for _ in sharp_crops_dir.rglob(f'*{ext}'))
            self.msg_queue.put(('log', f"Sharp crops directory: {sharp_crops_dir}"))
            self.msg_queue.put(('log', f"Found {image_count} images (recursive search)"))
        else:
            self.msg_queue.put(('log', "No sharp crops directory - using synthetic droplets only"))

        self.msg_queue.put(('status', "Generating synthetic blur data..."))
        self.msg_queue.put(('log', f"Generating {num_samples} samples..."))

        # Import and run generator
        try:
            # Force reload to pick up any code changes
            import sys
            import importlib
            if 'synthetic_blur' in sys.modules:
                importlib.reload(sys.modules['synthetic_blur'])

            from synthetic_blur import SyntheticBlurGenerator, BlurParams

            # Create optical params based on training mode
            if gui_training_mode == "direct":
                optical_params = BlurParams(
                    focal_length_mm=1.0,  # Placeholder — not used in direct mode
                    focus_distance_mm=1.0,
                    imaging_distance_mm=1.0,
                    aperture_diameter_mm=1.0,
                    pixel_size_mm=gui_pixel_size,
                    rho=1.0,
                    training_mode="direct",
                    rho_direct=gui_rho_direct,
                    sigma_0=gui_sigma_0,
                    scale_calib_px_per_mm=calib_scale_px_per_mm,
                )
                self.msg_queue.put(('log', f"Direct mode: rho_direct={gui_rho_direct} px/mm, "
                                   f"sigma_0={gui_sigma_0} px, "
                                    f"scale_calib={calib_scale_px_per_mm} px/mm"))
            else:
                optical_params = BlurParams(
                    focal_length_mm=gui_focal_length,
                    focus_distance_mm=gui_focus_distance,
                    imaging_distance_mm=gui_imaging_distance,
                    aperture_diameter_mm=gui_aperture_diameter,
                    pixel_size_mm=gui_pixel_size,
                    rho=gui_rho,
                )
                self.msg_queue.put(('log', f"Optical params: F={gui_focal_length}mm, "
                                   f"f/{gui_f_number}, d0={gui_focus_distance}mm"))
            self.msg_queue.put(('log', f"Defocus range: {gui_defocus_min} to {gui_defocus_max} mm"))

            original_size = min(config.sensor_width_px, config.sensor_height_px)

            if train_size != original_size:
                self.msg_queue.put(
                    ('log', f"Resizing: {original_size}×{original_size} → {train_size}×{train_size} px"))
            else:
                self.msg_queue.put(('log', f"Image size: {train_size}×{train_size} px"))

            # Get minimum blur if enabled
            min_blur_px = None
            if self.min_blur_enabled_var.get():
                try:
                    min_blur_px = float(self.min_blur_value_var.get())
                    if min_blur_px < 0:
                        min_blur_px = 0.0
                    _bt = "blur" if self.training_mode_var.get() == 'direct' else "CoC"
                    self.msg_queue.put(
                        ('log', f"Applying minimum {_bt} filter: {min_blur_px:.2f} px"))
                except ValueError:
                    _bt = "blur" if self.training_mode_var.get() == 'direct' else "CoC"
                    self.msg_queue.put(
                        ('log', f"Warning: Invalid min {_bt} value, ignoring filter"))
                    min_blur_px = None

            # Parse ERF validation count
            erf_validation_count = None
            if self.erf_validation_var.get():
                try:
                    count_str = self.erf_validation_count_var.get().strip()
                    if count_str:
                        erf_validation_count = int(count_str)
                        if erf_validation_count > num_samples:
                            self.msg_queue.put(
                                ('log',
                                 f"ERF validation count ({erf_validation_count}) > num_samples "
                                 f"({num_samples}), capping to {num_samples}"))
                            erf_validation_count = num_samples
                except ValueError:
                    self.msg_queue.put(
                        ('log', "Warning: Invalid ERF validation count, validating all samples"))
                    erf_validation_count = None

            generator = SyntheticBlurGenerator(
                optical_params=optical_params,
                defocus_range_mm=(gui_defocus_min, gui_defocus_max),
                image_size=train_size,
                crop_size=training_crop_size,                           # Actual crop size on disk
                calibration_reference_resolution=calib_reference_resolution,  # Resolution rho was measured at
                blur_distribution=self.blur_distribution_var.get(),
                beta_alpha=float(self.beta_alpha_var.get()),
                beta_beta=float(self.beta_beta_var.get()),
                min_blur_px=min_blur_px
            )

            # Data files (blur/, sharp/, blur_map/, metadata.csv) go directly into the dataset folder
            data_dir = output_dir

            camera_filter = self.camera_filter_var.get() if hasattr(self, 'camera_filter_var') else "all"

            gen_metadata = generator.generate_dataset(
                output_dir=data_dir, num_samples=num_samples, sharp_images_dir=sharp_crops_dir
                if(sharp_crops_dir and sharp_crops_dir.exists()) else None,
                diameter_range_px=(10, 200),
                camera_filter=camera_filter, save_blur_trace=self.save_blur_trace_var.get(),
                erf_validation=self.erf_validation_var.get(),
                erf_validation_count=erf_validation_count,)

            if gen_metadata['diameter_bins_used'] and gen_metadata['diameter_bin_boundaries'] is not None:
                config_path = output_dir / 'generation_config.yaml'
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)

                p33, p67 = gen_metadata['diameter_bin_boundaries']
                config_dict['data']['diameter_bins'] = {
                    'stratified': True,
                    'tertile_boundaries': [float(p33), float(p67)],
                    'bins': [
                        {'name': 'small', 'range': [f'<{p33:.1f}']},
                        {'name': 'medium', 'range': [f'{p33:.1f}-{p67:.1f}']},
                        {'name': 'large', 'range': [f'>{p67:.1f}']}
                    ]
                }

                with open(config_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)

                self.msg_queue.put(
                    ('log', f"Saved diameter bin boundaries to config: {p33:.1f}, {p67:.1f}"))

            # Count actual generated files
            actual_count = len(list((data_dir / 'blur').glob('*.png')))
            self.msg_queue.put(('log', f"Generated {actual_count} samples in {data_dir}"))
            self.msg_queue.put(('progress', 100))

            # Show blur distribution histogram (schedule on main thread)
            self.msg_queue.put(('show_histogram', str(data_dir)))

        except ImportError as e:
            self.msg_queue.put(
                ('error', f"Import error: {e}\nMake sure synthetic_blur.py is in the same directory."))

    def _show_coc_histogram(self, data_dir: Path):
        """Show histogram of blur distribution from metadata.csv."""
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            metadata_path = data_dir / 'metadata.csv'
            if not metadata_path.exists():
                self.msg_queue.put(('log', "⚠ metadata.csv not found, skipping histogram"))
                return

            # Read metadata
            df = pd.read_csv(metadata_path)
            blur_col = 'sigma_px' if 'sigma_px' in df.columns else 'coc_px'
            if blur_col not in df.columns:
                self.msg_queue.put(('log', "⚠ blur column not found in metadata.csv"))
                return

            blur_values = df[blur_col].abs().values
            total = len(blur_values)
            max_blur_val = blur_values.max()
            max_blur_ceil = int(np.ceil(max_blur_val))

            # Check if min blur was applied during generation
            min_blur_val = blur_values.min()

            # If min is significantly above 0, adjust bin range to use actual values
            if min_blur_val > 0.01:  # threshold to avoid numerical noise
                bin_range_start = min_blur_val
                max_blur_actual = max_blur_val
                bin_note = f"(bins shifted: [{min_blur_val:.2f}, {max_blur_val:.2f}])"
            else:
                bin_range_start = 0
                max_blur_actual = max_blur_val
                bin_note = f"(bins: [0, {max_blur_val:.2f}])"

            # Calculate binned statistics (4 equal bins from bin_range_start to max)
            bin_size = (max_blur_actual - bin_range_start) / 4.0
            bins_edges = [bin_range_start + i * bin_size for i in range(5)]  # 5 edges for 4 bins
            bin_labels = [f'{bins_edges[i]:.2f}-{bins_edges[i+1]:.2f}px' for i in range(4)]

            # Calculate bin weights from beta distribution
            bin_weights = self._calculate_bin_weights_from_beta()

            bin_counts = []
            for i in range(4):
                lower = bins_edges[i]
                upper = bins_edges[i + 1]
                if i == 3:  # Last bin includes max value
                    count = np.sum((blur_values >= lower) & (blur_values <= upper))
                else:
                    count = np.sum((blur_values >= lower) & (blur_values < upper))
                bin_counts.append(count)

            bin_percentages = [(count / total) * 100 for count in bin_counts]

            # Log statistics directly to log widget (since we're on main thread)
            _bt = "Blur" if self.training_mode_var.get() == 'direct' else "CoC"
            log_messages = []
            log_messages.append("\n" + "=" * 50)
            log_messages.append(f"{_bt} Distribution Summary:")
            log_messages.append(f"Total samples: {total}")
            log_messages.append(
                f"{_bt} Range: {min_blur_val:.2f} - {max_blur_val:.2f} px {bin_note}")
            log_messages.append(f"Mean {_bt}: {blur_values.mean():.2f} px")
            log_messages.append(f"Median {_bt}: {np.median(blur_values):.2f} px")
            log_messages.append("\nBinned Distribution:")
            for label, count, pct, target in zip(
                bin_labels, bin_counts, bin_percentages, bin_weights):
                target_pct = target * 100
                diff = pct - target_pct
                log_messages.append(
                    f"  {label}: {count:5d} ({pct:5.1f}%) | Target: {target_pct:5.1f}% | Diff: {diff:+5.1f}%")

            # Per-integer interval distribution
            log_messages.append(f"\nPer-Integer {_bt} Distribution:")
            for i in range(max_blur_ceil):
                lower = i
                upper = i + 1
                if i == max_blur_ceil - 1:  # Last interval includes max value
                    count = np.sum((blur_values >= lower) & (blur_values <= upper))
                else:
                    count = np.sum((blur_values >= lower) & (blur_values < upper))
                pct = (count / total) * 100
                bar = "█" * int(pct / 2)  # Scale for display
                log_messages.append(f"  {lower:2d}-{upper:2d} px: {count:5d} ({pct:5.1f}%) {bar}")
            log_messages.append("=" * 50 + "\n")

            # Print to console/terminal
            for msg in log_messages:
                print(msg)

            # Create histogram window
            hist_window = tk.Toplevel(self.root)
            hist_window.title(f"{_bt} Distribution")
            hist_window.geometry("1200x900")

            # Create figure with 3 subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

            # Top plot: Fine-grained histogram
            ax1.hist(blur_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax1.axvline(blur_values.mean(), color='red', linestyle='--',
                        linewidth=2, label=f'Mean: {blur_values.mean():.2f} px')
            ax1.axvline(np.median(blur_values), color='orange', linestyle='--',
                        linewidth=2, label=f'Median: {np.median(blur_values):.2f} px')
            ax1.set_xlabel(f'{_bt} (pixels)')
            ax1.set_ylabel('Count')
            ax1.set_title(f'{_bt} Distribution ({total} samples)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Middle plot: 4-bin actual vs target comparison
            x = np.arange(len(bin_labels))
            width = 0.35
            ax2.bar(x - width / 2, bin_percentages, width,
                    label='Actual', alpha=0.8, color='steelblue')
            ax2.bar(x + width / 2, [w * 100 for w in bin_weights],
                    width, label='Target', alpha=0.8, color='orange')

            ax2.set_xlabel(f'{_bt} Bin')
            ax2.set_ylabel('Percentage (%)')
            ax2.set_title('Actual vs Target Distribution (4-Bin)')
            ax2.set_xticks(x)
            ax2.set_xticklabels(bin_labels)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')

            # Add percentage labels on bars
            max_pct_4bin = max(max(bin_percentages), max([w * 100 for w in bin_weights]))
            for i, (actual, target) in enumerate(
                zip(bin_percentages, [w * 100 for w in bin_weights])):
                ax2.text(i - width / 2, actual + 1,
                         f'{actual:.1f}%', ha='center', va='bottom', fontsize=9)
                ax2.text(i + width / 2, target + 1,
                         f'{target:.1f}%', ha='center', va='bottom', fontsize=9)

            # Set y-limit to ensure labels fit (add 15% padding)
            ax2.set_ylim(0, max_pct_4bin * 1.15)

            # Bottom plot: Per-integer interval histogram
            integer_counts = []
            integer_labels = []

            for i in range(max_blur_ceil):
                lower = i
                upper = i + 1
                if i == max_blur_ceil - 1:  # Last interval includes max value
                    count = np.sum((blur_values >= lower) & (blur_values <= upper))
                else:
                    count = np.sum((blur_values >= lower) & (blur_values < upper))
                integer_counts.append(count)
                integer_labels.append(f'{lower}-{upper}')

            # Convert to percentages
            integer_percentages = [(count / total) * 100 for count in integer_counts]

            # Create bar chart
            x_int = np.arange(len(integer_labels))
            ax3.bar(x_int, integer_percentages, alpha=0.8, color='steelblue', edgecolor='black')
            ax3.set_xlabel(f'{_bt} Interval (pixels)')
            ax3.set_ylabel('Percentage (%)')
            ax3.set_title('Per-Integer Interval Distribution')
            ax3.set_xticks(x_int)
            ax3.set_xticklabels(integer_labels, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3, axis='y')

            # Add percentage labels on bars (only if not too many)
            if len(integer_labels) <= 20:
                max_pct_int = max(integer_percentages) if integer_percentages else 0
                for i, pct in enumerate(integer_percentages):
                    if pct > 0.5:  # Only show if > 0.5%
                        ax3.text(i, pct + 0.3, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

                # Set y-limit to ensure labels fit (add 15% padding)
                if max_pct_int > 0:
                    ax3.set_ylim(0, max_pct_int * 1.15)

            # Add vertical spacing between subplots
            plt.tight_layout(h_pad=3.0)

            # Save figure to file
            output_path = data_dir / 'blur_distribution.png'
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            self.msg_queue.put(('log', f"Saved histogram to: {output_path}"))

            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, master=hist_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

            # Add close button
            close_btn = ttk.Button(hist_window, text="Close", command=hist_window.destroy)
            close_btn.pack(pady=5)

        except Exception as e:
            self.msg_queue.put(('log', f"⚠ Error showing histogram: {e}"))

    def _train_model(self):
        """Train the model."""
        if not self._validate_paths():
            return

        output_dir = Path(self.output_dir_var.get())
        data_dir = output_dir / 'synthetic_data'

        if not data_dir.exists():
            messagebox.showwarning(
                "Warning", "Synthetic data not found. Please generate data first.")
            return

        self._set_training_state(True)
        self._log("\n" + "=" * 50)
        self._log("Starting model training...")

        def train_thread():
            try:
                self._run_training()
                self.msg_queue.put(('training_complete', None))
            except Exception as e:
                import traceback
                error_msg = f"{e}\n{traceback.format_exc()}"
                print(f"\nERROR: {error_msg}", flush=True)  # Echo to terminal
                self.msg_queue.put(('error', error_msg))

        self.training_thread = threading.Thread(target=train_thread, daemon=True)
        self.training_thread.start()

    def _train_dme_only(self):
        """Train only the DME subnet."""
        if not self._validate_paths():
            return

        output_dir = Path(self.output_dir_var.get())
        data_dir = output_dir / 'synthetic_data'

        if not data_dir.exists():
            messagebox.showwarning(
                "Warning", "Synthetic data not found. Please generate data first.")
            return

        self._set_training_state(True)
        self._log("\n" + "=" * 50)
        self._log("Starting DME-only training...")

        def train_thread():
            try:
                self._run_dme_only_training()
                self.msg_queue.put(('training_complete', None))
            except Exception as e:
                import traceback
                error_msg = f"{e}\n{traceback.format_exc()}"
                print(f"\nERROR: {error_msg}", flush=True)  # Echo to terminal
                self.msg_queue.put(('error', error_msg))

        self.training_thread = threading.Thread(target=train_thread, daemon=True)
        self.training_thread.start()

    def _run_dme_only_training(self):
        """Run DME-only training (in thread)."""
        # Set CUDA launch blocking if requested (must be done before any CUDA operations)
        if self.cuda_launch_blocking_var.get():
            import os
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            self.msg_queue.put(
                ('log', "⚠ CUDA_LAUNCH_BLOCKING enabled - training will be slower but errors will be precise"))

        resolved = self._resolve_training_paths()
        if resolved is None:
            return
        data_dir, run_dir, config = resolved
        self._apply_gui_training_settings(config)

        self.msg_queue.put(('status', "Initialising trainer..."))

        try:
            from train import Trainer

            device = 'cuda' if self.use_gpu_var.get() else 'cpu'

            trainer = Trainer(
                config=config,
                data_dir=data_dir,
                output_dir=run_dir,
                device=device,
                stop_flag=lambda: self.stop_training
            )
            trainer.write_run_metadata(status='started', run_name=self.run_name_var.get().strip() or None)

            self.msg_queue.put(('log', f"Training on device: {trainer.device}"))
            self.msg_queue.put(('log', f"Run folder: {run_dir}"))
            self.msg_queue.put(('status', "Training DME-subnet..."))

            checkpoint_value = self.checkpoint_path_var.get()
            force_fresh = (checkpoint_value == "")
            explicit_checkpoint = checkpoint_value if checkpoint_value else None

            trainer.train_dme_only(
                checkpoint_preference='best',
                explicit_checkpoint=explicit_checkpoint,
                force_fresh_start=force_fresh
            )

            self.msg_queue.put(('log', "DME training complete!"))
            self.msg_queue.put(('log', f"Model saved to: {run_dir / 'checkpoints' / 'dme_best.pth'}"))

        except ImportError as e:
            self.msg_queue.put(('error', f"Import error: {e}"))

    def _run_training(self):
        """Run model training (in thread)."""
        # Set CUDA launch blocking if requested (must be done before any CUDA operations)
        if self.cuda_launch_blocking_var.get():
            import os
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            self.msg_queue.put(
                ('log', "⚠ CUDA_LAUNCH_BLOCKING enabled - training will be slower but errors will be precise"))

        resolved = self._resolve_training_paths()
        if resolved is None:
            return
        data_dir, run_dir, config = resolved
        self._apply_gui_training_settings(config)

        self.msg_queue.put(('status', "Initialising trainer..."))

        try:
            from train import Trainer

            device = 'cuda' if self.use_gpu_var.get() else 'cpu'

            trainer = Trainer(
                config=config,
                data_dir=data_dir,
                output_dir=run_dir,
                device=device,
                stop_flag=lambda: self.stop_training
            )
            trainer.write_run_metadata(status='started', run_name=self.run_name_var.get().strip() or None)

            self.msg_queue.put(('log', f"Training on device: {trainer.device}"))
            self.msg_queue.put(('log', f"Run folder: {run_dir}"))
            self.msg_queue.put(('status', "Training DME..."))

            checkpoint_value = self.checkpoint_path_var.get()
            explicit_checkpoint = None if checkpoint_value == "" else checkpoint_value

            trainer.train(resume_from=explicit_checkpoint)

            self.msg_queue.put(('log', "Training complete!"))
            self.msg_queue.put(('log', f"Model saved to: {run_dir / 'checkpoints' / 'dme_best.pth'}"))

        except ImportError as e:
            self.msg_queue.put(('error', f"Import error: {e}"))

    # =========================================================================
    # Inference
    # =========================================================================
    def _run_inference(self):
        """Run inference on real crops."""
        # Validate paths
        model_path = self.inf_model_var.get()
        input_dir = self.inf_input_var.get()
        output_dir = self.inf_output_var.get()

        if not model_path or not Path(model_path).exists():
            messagebox.showerror("Error", "Please select a valid model checkpoint.")
            return

        if not input_dir or not Path(input_dir).exists():
            messagebox.showerror("Error", "Please select a valid input directory.")
            return

        if not output_dir:
            messagebox.showerror("Error", "Please specify an output directory.")
            return

        # Disable button, enable stop button
        self.run_inference_btn.config(state='disabled')
        self.stop_inference_btn.config(state='normal')
        self.inf_status_var.set("Starting inference...")
        self.inf_progress_var.set(0)

        # Clear previous results
        self.inf_results_text.config(state='normal')
        self.inf_results_text.delete('1.0', tk.END)
        self.inf_results_text.config(state='disabled')

        # Run in thread
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()

    def _inference_worker(self):
        """Worker thread for inference."""
        try:
            import sys
            from datetime import datetime
            sys.path.insert(0, str(Path(__file__).parent))

            from inference_real_crops import RealCropInference

            model_path = self.inf_model_var.get()
            input_dir = Path(self.inf_input_var.get())

            # Use existing run dir from preprocessing if available, else create new one
            if hasattr(self, '_inf_run_dir') and self._inf_run_dir:
                output_dir = Path(self._inf_run_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                base_output = Path(self.inf_output_var.get())
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = base_output / f"inference_{timestamp}"
                output_dir.mkdir(parents=True, exist_ok=True)

            device = 'cuda' if self.inf_use_gpu_var.get() else 'cpu'

            self.msg_queue.put(('log', f"\n{'='*60}"))
            self.msg_queue.put(('log', "INFERENCE ON REAL CROPS"))
            self.msg_queue.put(('log', f"{'='*60}"))
            self.msg_queue.put(('log', f"Model: {model_path}"))
            self.msg_queue.put(('log', f"Input: {input_dir}"))
            self.msg_queue.put(('log', f"Output: {output_dir}"))
            self.msg_queue.put(('log', f"Device: {device}"))

            # Create inference engine
            self.msg_queue.put(('status', "Loading model..."))

            # Read inference camera scale for cross-camera correction (direct mode)
            inf_camera_scale = None
            inf_camera_scale_str = self.inf_camera_scale_var.get().strip()
            if inf_camera_scale_str:
                try:
                    inf_camera_scale = float(inf_camera_scale_str)
                    self.msg_queue.put(
                        ('log', f"Inference camera scale: {inf_camera_scale:.2f} px/mm"))
                except ValueError:
                    self.msg_queue.put(
                        ('log', f"WARNING: Invalid inference camera scale '{inf_camera_scale_str}' — ignoring"))

            inference = RealCropInference(
                model_path=str(model_path),
                device=device,
                inference_camera_scale_px_per_mm=inf_camera_scale,
            )

            # Log the loaded model's key parameters
            self.msg_queue.put(('log', f"\nModel loaded — mode: {inference.training_mode}"))
            if inference.training_mode == 'direct' and inference.direct_slope is not None:
                training_cfg = inference.config.get('training', {})
                scale_calib = training_cfg.get('scale_calib_px_per_mm')
                self.msg_queue.put(('log', f"  rho_eff: {inference.direct_slope:.6f} px/mm"))
                self.msg_queue.put(('log', f"  scale_calib: {scale_calib} px/mm"))
                if inf_camera_scale is not None:
                    self.msg_queue.put(('log', f"  scale_inf: {inf_camera_scale:.2f} px/mm"))
                else:
                    self.msg_queue.put(
                        ('log', f"  scale_inf: not set (no cross-camera correction)"))
                self.msg_queue.put(('log', f"  model_size: {inference.model_size} px"))
                self.msg_queue.put(('log', f"  max_sigma: {inference.max_blur:.4f} px"))
                self.msg_queue.put(
                    ('log',
                     f"  Inversion: |z| = sigma_model × native_size / ({inference.direct_slope:.4f} × {inference.model_size})"))

            # Get options
            save_viz = self.inf_save_viz_var.get()
            viz_rate = int(self.inf_viz_rate_var.get()) if self.inf_viz_rate_var.get() else 10

            self.msg_queue.put(('log', f"\nOptions:"))
            self.msg_queue.put(
                ('log', f"  Save visualizations: {save_viz} (1 per {viz_rate} crops)"))

            # Find material folders OR check for images directly in input_dir
            material_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

            # Check if there are images directly in the input folder (no subfolders case)
            direct_crops = list(input_dir.glob('*_crop.png'))
            direct_images = list(input_dir.glob(
                '*.png')) if len(direct_crops) == 0 else direct_crops

            self.msg_queue.put(('log', f"\nScanning input directory: {input_dir}"))
            self.msg_queue.put(('log', f"  Subfolders found: {len(material_dirs)}"))
            self.msg_queue.put(('log', f"  *_crop.png files: {len(direct_crops)}"))
            self.msg_queue.put(('log', f"  *.png files: {len(direct_images)}"))

            if len(material_dirs) == 0 and len(direct_images) == 0:
                self.msg_queue.put(
                    ('error', f"No material subdirectories or images found in {input_dir}"))
                return

            # Decide processing mode
            if len(material_dirs) > 0:
                # Standard mode: process subfolders as materials
                self.msg_queue.put(('log', f"\nFound {len(material_dirs)} material folders:"))
                for d in material_dirs:
                    crop_files = list(d.glob('*_crop.png'))
                    if len(crop_files) == 0:
                        crop_files = list(d.glob('*.png'))
                    self.msg_queue.put(('log', f"  - {d.name}: {len(crop_files)} images"))

                # Process each material
                all_results = []
                total_materials = len(material_dirs)

                for idx, material_dir in enumerate(material_dirs):
                    progress = (idx / total_materials) * 90  # Reserve 10% for final analysis
                    self.msg_queue.put(('inf_progress', progress))

                    self.msg_queue.put(('status', f"Processing {material_dir.name}..."))

                    df = inference.process_material_folder(
                        material_dir=material_dir,
                        output_base=output_dir,

                        save_visualizations=save_viz,
                        viz_sample_rate=viz_rate
                    )

                    if len(df) > 0:
                        all_results.append(df)
            else:
                # Single folder mode: treat input_dir itself as the material folder
                self.msg_queue.put(
                    ('log', f"\nNo subfolders found - processing images directly from input folder"))
                self.msg_queue.put(('log', f"  Found {len(direct_images)} images"))

                all_results = []
                self.msg_queue.put(('status', f"Processing {input_dir.name}..."))

                df = inference.process_material_folder(
                    material_dir=input_dir,
                    output_base=output_dir,
                    save_visualizations=save_viz,
                    viz_sample_rate=viz_rate
                )

                if len(df) > 0:
                    all_results.append(df)

            # Combine and create summary
            if len(all_results) > 0:
                import pandas as pd

                self.msg_queue.put(('status', "Creating summary analysis..."))
                self.msg_queue.put(('inf_progress', 95))

                combined_df = pd.concat(all_results, ignore_index=True)

                # Save combined CSV
                summary_path = output_dir / 'summary_all_materials.csv'
                combined_df.to_csv(summary_path, index=False)
                self.msg_queue.put(('log', f"\n✓ Saved combined summary to: {summary_path}"))

                # Create summary plots
                inference._create_summary_plots(combined_df, output_dir)

                # Prepare results summary for display
                summary_text = f"\n{'='*60}\n"
                summary_text += "OVERALL STATISTICS\n"
                summary_text += f"{'='*60}\n"
                summary_text += f"Total crops processed: {len(combined_df)}\n"
                summary_text += f"Materials: {combined_df['material'].nunique()}\n\n"

                # Focus status summary
                if 'focus_status' in combined_df.columns:
                    in_focus = (combined_df['focus_status'] == 'in_focus').sum()
                    out_of_focus = (combined_df['focus_status'] == 'out_of_focus').sum()
                    summary_text += "Focus Status:\n"
                    summary_text += f"  In Focus:      {in_focus:,} ({in_focus/len(combined_df)*100:.1f}%)\n"
                    summary_text += f"  Out of Focus:  {out_of_focus:,} ({out_of_focus/len(combined_df)*100:.1f}%)\n\n"

                blur_col = 'sigma_px' if 'sigma_px' in combined_df.columns else 'coc_px'
                blur_label = "Sigma" if blur_col == 'sigma_px' else "CoC"
                summary_text += f"{blur_label} Statistics (all materials):\n"
                summary_text += f"  Mean:   {combined_df[blur_col].mean():.2f} px\n"
                summary_text += f"  Median: {combined_df[blur_col].median():.2f} px\n"
                summary_text += f"  Std:    {combined_df[blur_col].std():.2f} px\n"
                summary_text += f"  Range:  {combined_df[blur_col].min():.2f} - {combined_df[blur_col].max():.2f} px\n\n"

                summary_text += "Defocus Statistics (all materials):\n"
                summary_text += f"  Mean:   {combined_df['defocus_mm'].mean():.2f} mm\n"
                summary_text += f"  Median: {combined_df['defocus_mm'].median():.2f} mm\n"
                summary_text += f"  Range:  {combined_df['defocus_mm'].min():.2f} - {combined_df['defocus_mm'].max():.2f} mm\n\n"

                # Per-material breakdown
                summary_text += "=" * 60 + "\n"
                summary_text += "BY MATERIAL:\n"
                summary_text += "=" * 60 + "\n"
                for material in combined_df['material'].unique():
                    mat_df = combined_df[combined_df['material'] == material]
                    summary_text += f"\n{material}:\n"
                    summary_text += f"  Crops: {len(mat_df)}\n"
                    summary_text += f"  {blur_label} Mean: {mat_df[blur_col].mean():.2f} px (±{mat_df[blur_col].std():.2f})\n"
                    summary_text += f"  Defocus: {mat_df['defocus_mm'].mean():.2f} mm (±{mat_df['defocus_mm'].std():.2f})\n"

                summary_text += "\n" + "=" * 60 + "\n"
                summary_text += f"✓ Results saved to: {output_dir}\n"
                summary_text += f"✓ Summary plots: {output_dir / 'summary_analysis.png'}\n"
                summary_text += "=" * 60

                # Display in results text box
                self.msg_queue.put(('inf_results', summary_text))

                self.msg_queue.put(('log', summary_text))

            self.msg_queue.put(('inf_progress', 100))
            self.msg_queue.put(('status', "Inference complete!"))
            self.msg_queue.put(('inference_complete', None))

        except Exception as e:
            import traceback
            error_msg = f"Inference error: {str(e)}\n{traceback.format_exc()}"
            self.msg_queue.put(('error', error_msg))
            self.msg_queue.put(('inference_complete', None))

    def _stop_inference(self):
        """Stop inference (placeholder - inference doesn't support stopping yet)."""
        self.msg_queue.put(('log', "⚠ Stopping inference not yet implemented"))
        # Future: Add a stop flag that the inference worker checks

    def _run_validation(self):
        """Run synthetic validation test."""
        # Validate paths
        model_path = self.inf_model_var.get()
        input_dir = self.inf_input_var.get()
        output_dir = self.inf_output_var.get()

        if not model_path or not Path(model_path).exists():
            messagebox.showerror("Error", "Please select a valid model checkpoint.")
            return

        if not input_dir or not Path(input_dir).exists():
            messagebox.showerror("Error", "Please select a valid input directory.")
            return

        # Ask for validation settings
        from tkinter import simpledialog

        num_samples = simpledialog.askinteger(
            "Validation Settings",
            "How many crops to test?\n(More = longer but more reliable)",
            initialvalue=20,
            minvalue=5,
            maxvalue=100
        )

        if not num_samples:
            return

        coc_values_str = simpledialog.askstring(
            "Validation Settings",
            "CoC values to test (space-separated, in pixels):",
            initialvalue="2.0 5.0 10.0 15.0"
        )

        if not coc_values_str:
            return

        try:
            coc_values = [float(x) for x in coc_values_str.split()]
        except ValueError:
            messagebox.showerror("Error", "Invalid CoC values. Use space-separated numbers.")
            return

        # Disable buttons
        self.run_inference_btn.config(state='disabled')
        self.run_validation_btn.config(state='disabled')
        self.stop_inference_btn.config(state='normal')
        self.inf_status_var.set("Running validation...")
        self.inf_progress_var.set(0)

        # Clear previous results
        self.inf_results_text.config(state='normal')
        self.inf_results_text.delete('1.0', tk.END)
        self.inf_results_text.config(state='disabled')

        # Run in thread
        self.validation_thread = threading.Thread(
            target=self._validation_worker,
            args=(num_samples, coc_values),
            daemon=True
        )
        self.validation_thread.start()

    def _validation_worker(self, num_samples: int, coc_values: List[float]):
        """Worker thread for validation."""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))

            from validation_synthetic import SyntheticValidator

            model_path = self.inf_model_var.get()
            input_dir = Path(self.inf_input_var.get())
            output_dir = Path(self.inf_output_var.get()) / 'validation'

            device = 'cuda' if self.inf_use_gpu_var.get() else 'cpu'

            self.msg_queue.put(('log', f"\n{'='*60}"))
            self.msg_queue.put(('log', "SYNTHETIC VALIDATION TEST"))
            self.msg_queue.put(('log', f"{'='*60}"))
            self.msg_queue.put(('log', f"Model: {model_path}"))
            self.msg_queue.put(('log', f"Crops: {input_dir}"))
            self.msg_queue.put(('log', f"Test CoC values: {coc_values}"))
            self.msg_queue.put(('log', f"Number of samples: {num_samples}"))
            self.msg_queue.put(('log', f"Device: {device}"))

            # Create validator
            self.msg_queue.put(('status', "Loading model for validation..."))
            self.msg_queue.put(('inf_progress', 10))

            validator = SyntheticValidator(
                model_path=str(model_path),
                device=device
            )

            # Pick folder with crops - either subfolder or input_dir itself
            material_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

            if len(material_dirs) > 0:
                # Use first material subfolder
                test_dir = material_dirs[0]
                self.msg_queue.put(('log', f"\nUsing crops from subfolder: {test_dir.name}"))
            else:
                # No subfolders - use input_dir directly (flat folder mode)
                direct_images = list(input_dir.glob('*_crop.png'))
                if len(direct_images) == 0:
                    direct_images = list(input_dir.glob('*.png'))
                if len(direct_images) == 0:
                    self.msg_queue.put(('error', f"No images found in {input_dir}"))
                    return
                test_dir = input_dir
                self.msg_queue.put(
                    ('log', f"\nUsing crops directly from: {test_dir.name} ({len(direct_images)} images)"))
            self.msg_queue.put(('inf_progress', 20))

            # Run validation
            self.msg_queue.put(('status', "Running validation tests..."))

            df = validator.validate_on_crops(
                crops_dir=test_dir,
                test_coc_values=coc_values,
                num_samples=num_samples,
                output_dir=output_dir
            )

            self.msg_queue.put(('inf_progress', 90))

            if len(df) > 0:
                # Prepare results summary
                summary_text = f"\n{'='*60}\n"
                summary_text += "VALIDATION RESULTS\n"
                summary_text += f"{'='*60}\n"
                summary_text += f"Total tests: {len(df)}\n\n"

                summary_text += "Blur Estimation:\n"
                summary_text += f"  MAE:    {df['error_px'].mean():.2f} px\n"
                summary_text += f"  RMSE:   {np.sqrt((df['error_px']**2).mean()):.2f} px\n"
                summary_text += f"  Median: {df['error_px'].median():.2f} px\n\n"

                summary_text += "Diameter Recovery:\n"
                summary_text += f"  MAE:          {df['diameter_error_px'].mean():.2f} px\n"
                summary_text += f"  Mean Error %: {df['diameter_error_pct'].mean():.2f}%\n"
                summary_text += f"  Median Error: {df['diameter_error_px'].median():.2f} px\n\n"

                summary_text += "Per-Blur Performance:\n"
                for coc_val in coc_values:
                    gt_col = 'sigma_gt_px' if 'sigma_gt_px' in df.columns else 'coc_gt_px'
                    subset = df[df[gt_col] == coc_val]
                    if len(subset) > 0:
                        summary_text += f"\n  Blur = {coc_val} px:\n"
                        summary_text += f"    Blur Error:     {subset['error_px'].mean():.2f} px\n"
                        summary_text += f"    Diameter Error: {subset['diameter_error_px'].mean():.2f} px "
                        summary_text += f"({subset['diameter_error_pct'].mean():.2f}%)\n"

                summary_text += "\n" + "=" * 60 + "\n"
                summary_text += f"✓ Results saved to: {output_dir}\n"
                summary_text += f"✓ Plots: {output_dir / 'validation_analysis.png'}\n"
                summary_text += f"✓ CSV: {output_dir / 'synthetic_validation_results.csv'}\n"
                summary_text += "=" * 60

                # Display in results text box
                self.msg_queue.put(('inf_results', summary_text))
                self.msg_queue.put(('log', summary_text))

            self.msg_queue.put(('inf_progress', 100))
            self.msg_queue.put(('status', "Validation complete!"))
            self.msg_queue.put(('inference_complete', None))

        except Exception as e:
            import traceback
            error_msg = f"Validation error: {str(e)}\n{traceback.format_exc()}"
            self.msg_queue.put(('error', error_msg))
            self.msg_queue.put(('inference_complete', None))

    # =========================================================================
    # Validation Testing (Tab 5) - New Methods
    # =========================================================================
    def _run_validation_test(self):
        """Run validation/test based on selected mode (Tab 5)."""
        # Validate paths
        model_path = self.val_model_var.get()
        data_path = self.val_data_var.get()
        output_path = self.val_output_var.get()

        if not model_path or not Path(model_path).exists():
            messagebox.showerror("Error", "Please select a valid model checkpoint.")
            return

        if not data_path or not Path(data_path).exists():
            messagebox.showerror("Error", "Please select a valid test data directory.")
            return

        # Disable buttons
        self.run_val_btn.config(state='disabled')
        self.stop_val_btn.config(state='normal')
        self.val_status_var.set("Starting test...")
        self.val_progress_var.set(0)

        # Clear previous results
        self.val_results_text.config(state='normal')
        self.val_results_text.delete('1.0', tk.END)
        self.val_results_text.config(state='disabled')

        # Run in thread
        self.validation_test_thread = threading.Thread(
            target=self._validation_test_worker, daemon=True)
        self.validation_test_thread.start()

    def _validation_test_worker(self):
        """Worker thread for Tab 5 validation/testing."""
        mode = self.val_mode_var.get()

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))

            model_path = self.val_model_var.get()
            data_path = Path(self.val_data_var.get())
            output_path = Path(self.val_output_var.get())
            device = 'cuda' if self.val_use_gpu_var.get() else 'cpu'

            # Get performance settings
            batch_size = int(self.val_batch_size_var.get())
            num_workers = int(self.val_num_workers_var.get())

            # Get visualization percentage
            viz_percent = float(self.val_viz_percent_var.get())

            # Calculate num_samples based on percentage or number selection
            sample_method = self.val_sample_method_var.get()
            if sample_method == "percentage":
                # Count files in synthetic_data
                blur_dir = data_path / 'blur'
                total_files = len(list(blur_dir.glob('*.png'))) if blur_dir.exists() else 0
                percentage = int(self.val_percentage_var.get())
                num_samples = max(1, int(total_files * percentage / 100.0))
            else:
                num_samples = int(self.val_num_samples_var.get()
                                  ) if self.val_num_samples_var.get() else 0

            self.msg_queue.put(('log', f"\n{'='*60}"))
            self.msg_queue.put(('log', f"MODEL TEST: {mode.upper()}"))
            self.msg_queue.put(('log', f"{'='*60}"))

            if mode == "dme":
                # DME test using test_model.py logic
                from test_model import ModelTester

                self.msg_queue.put(('val_status', 'Running DME test...'))
                self.msg_queue.put(('val_progress', 10))

                tester = ModelTester(model_path=str(model_path), device=device)

                # Get worst-case settings
                num_worst_px = int(self.val_num_worst_var.get()
                                   ) if self.val_save_worst_var.get() else 0
                num_worst_pct = int(self.val_num_worst_var_2.get()
                                    ) if self.val_save_worst_var_2.get() else 0

                # Get blur filter settings
                enable_coc_filter = self.val_enable_coc_filter_var.get()
                min_blur_threshold = float(self.val_min_blur_var.get()
                                           ) if enable_coc_filter else None
                filter_worst_pct = self.val_filter_worst_pct_var.get() if enable_coc_filter else False
                filter_metrics = self.val_filter_metrics_var.get() if enable_coc_filter else False
                exclude_from_test = self.val_exclude_from_test_var.get() if enable_coc_filter else False
                # Note: filter_intervals and filter_plots removed from GUI - always use defaults (False)

                df = tester.test_dme_only(
                    data_dir=str(data_path),
                    num_samples=num_samples,
                    output_dir=str(output_path),
                    viz_percent=viz_percent,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    num_worst_px=num_worst_px,
                    num_worst_pct=num_worst_pct,
                    min_blur_filter=min_blur_threshold,
                    filter_worst_pct=filter_worst_pct,
                    filter_metrics=filter_metrics,
                    exclude_from_test=exclude_from_test,
                    filter_intervals=False,
                    filter_plots=False
                )

                if len(df) > 0:
                    # Compute binned MAE for weighted metric
                    # Use actual min/max from config to match training bins
                    min_blur = float(self.val_min_blur_var.get())
                    max_blur = tester.max_blur
                    blur_range = max_blur - min_blur
                    bin_size = blur_range / 4.0
                    bins = [(min_blur + i * bin_size, min_blur + (i + 1) * bin_size)
                             for i in range(4)]

                    # Use bin weights from the tester (calculated from model's training config)
                    bin_weights = tester.bin_weights
                    bin_maes = []

                    for low, high in bins:
                        mask = (df[f'{tester.blur_col}_gt_px'] >=low) &(
                            df[f'{tester.blur_col}_gt_px'] <high)
                        bin_errors = df[mask]['error_px'].values
                        bin_maes.append(np.mean(bin_errors) if len(bin_errors) > 0 else 0.0)

                    weighted_mae = sum(w * m for w, m in zip(bin_weights, bin_maes))

                    # Format weights as percentages for display
                    weights_str = '-'.join([f"{int(w*100)}" for w in bin_weights])

                    # Defocus distance metrics
                    defocus_mae_mm = df['defocus_error_mm'].mean()
                    defocus_rmse_mm = np.sqrt((df['defocus_error_mm']**2).mean())

                    summary = f"\n{'='*60}\nDME TEST RESULTS\n{'='*60}\n"
                    summary += f"Samples: {len(df)}\n\n"
                    summary += f"{tester.blur_term} Metrics:\n"
                    summary += f"  Uniform MAE:  {df['error_px'].mean():.2f} px\n"
                    summary += f"  Weighted MAE: {weighted_mae:.2f} px ({weights_str}% weighting)\n"
                    summary += f"  RMSE: {np.sqrt((df['error_px']**2).mean()):.2f} px\n\n"
                    summary += f"Defocus Distance Metrics:\n"
                    summary += f"  MAE:  {defocus_mae_mm:.2f} mm\n"
                    summary += f"  RMSE: {defocus_rmse_mm:.2f} mm\n"

                    # Show calibration uncertainty if available
                    if 'defocus_uncertainty_mm' in df.columns:
                        mean_unc = df['defocus_uncertainty_mm'].mean()
                        if mean_unc > 0:
                            summary += f"\nCalibration Uncertainty:\n"
                            summary += f"  Mean: \u00b1{mean_unc:.3f} mm\n"
                            # Compare model error to calibration uncertainty
                            if defocus_mae_mm < mean_unc:
                                summary += f"  Model error < calibration uncertainty (at floor)\n"
                            else:
                                summary += f"  Model error > calibration uncertainty (room to improve)\n"

                    summary += f"\n\u2713 Results: {output_path}\n{'='*60}"
                    self.msg_queue.put(('val_results', summary))
                    self.msg_queue.put(('log', summary))

            self.msg_queue.put(('val_progress', 100))
            self.msg_queue.put(('val_status', 'Test complete'))
            self.msg_queue.put(('validation_test_complete', None))

        except Exception as e:
            import traceback
            error_msg = f"Test error: {str(e)}\n{traceback.format_exc()}"
            self.msg_queue.put(('error', error_msg))
            self.msg_queue.put(('val_status', 'Test failed'))
            self.msg_queue.put(('validation_test_complete', None))

    def _stop_validation(self):
        """Stop validation test."""
        self.msg_queue.put(('log', "⚠ Stopping test..."))
        # Future: Add stop flag

    def _run_full_pipeline(self):
        """Run full pipeline: generate + train."""
        if not self._validate_paths():
            return

        if not self.folder_configs:
            messagebox.showwarning(
                "Warning", "No optical configs saved. Please scan and configure first.")
            return

        self._set_training_state(True)
        self._log("\n" + "=" * 50)
        self._log("Starting full training pipeline...")

        def full_thread():
            try:
                self._run_generation()
                if not self.stop_training:
                    self._run_training()
                self.msg_queue.put(('training_complete', None))
            except Exception as e:
                import traceback
                error_msg = f"{e}\n{traceback.format_exc()}"
                print(f"\nERROR: {error_msg}", flush=True)  # Echo to terminal
                self.msg_queue.put(('error', error_msg))

        self.training_thread = threading.Thread(target=full_thread, daemon=True)
        self.training_thread.start()

    def _stop_training(self):
        """Stop training."""
        self.stop_training = True
        self._log("⚠️  Stopping training after current batch completes...")
        # Note: Don't call _set_training_state(False) here - let training thread do it when it exits

    def _validate_paths(self) -> bool:
        """Validate required paths are set and mode-specific requirements."""
        if not self.output_dir_var.get():
            messagebox.showwarning("Warning", "Please set output directory.")
            return False

        # USER CONSTRAINT: Validate direct mode has calibration loaded
        mode = self.training_mode_var.get()

        if mode == "direct":
            # Check if calibration file was loaded
            if not self.rho_direct_var.get() or not self.sigma_0_var.get():
                messagebox.showerror(
                    "Configuration Error",
                    "Direct Calibration mode requires a calibration file.\n\n"
                    "Click 'Browse...' in the Direct Calibration Parameters section "
                    "to load calibration data before starting training.")
                return False

            # Verify values are valid (redundant but defensive)
            try:
                rho_direct = float(self.rho_direct_var.get())
                sigma_0 = float(self.sigma_0_var.get())

                if rho_direct <= 0 or sigma_0 < 0:
                    raise ValueError("Invalid calibration parameters")
            except (ValueError, AttributeError):
                messagebox.showerror("Configuration Error",
                                     "Invalid direct calibration parameters.\n\n"
                                     "Please reload the calibration file.")
                return False

            self._log("Validation passed: Direct mode calibration parameters loaded")

        return True

    def _set_training_state(self, is_training: bool):
        """Update UI state during training."""
        state = 'disabled' if is_training else 'normal'
        self.generate_btn.config(state=state)
        self.mode_full_btn.config(state=state)
        self.mode_dme_btn.config(state=state)
        self.start_train_btn.config(state=state)
        self.stop_btn.config(state='normal' if is_training else 'disabled')

        if not is_training:
            self.stop_training = False

    # =========================================================================
    # Logging & Messages
    # =========================================================================
    def _log(self, message: str):
        """Add message to log."""
        self.msg_queue.put(('log', message))

    def _process_messages(self):
        """Process messages from worker threads."""
        try:
            while True:
                msg_type, data = self.msg_queue.get_nowait()

                if msg_type == 'log':
                    # Log widget removed - print to terminal instead
                    print(data, flush=True)

                elif msg_type == 'status':
                    self.status_var.set(data)

                elif msg_type == 'progress':
                    self.progress_var.set(data)

                elif msg_type == 'scan_complete':
                    self._on_scan_complete(data)

                elif msg_type == 'generation_complete':
                    self._log("Data generation complete!")
                    self.status_var.set("Generation complete")
                    self._set_training_state(False)

                elif msg_type == 'show_histogram':
                    # Show histogram on main thread
                    self._show_coc_histogram(Path(data))

                elif msg_type == 'training_complete':
                    self._log("Training pipeline complete!")
                    self.status_var.set("Complete")
                    self.progress_var.set(100)
                    self._set_training_state(False)
                    messagebox.showinfo("Complete", "Training pipeline finished!")

                elif msg_type == 'error':
                    self._log(f"ERROR: {data}")
                    self.status_var.set("Error")
                    self._set_training_state(False)
                    messagebox.showerror("Error", str(data))

                elif msg_type == 'inf_progress':
                    self.inf_progress_var.set(data)

                elif msg_type == 'inf_results':
                    self.inf_results_text.config(state='normal')
                    self.inf_results_text.delete('1.0', tk.END)
                    self.inf_results_text.insert('1.0', data)
                    self.inf_results_text.config(state='disabled')

                elif msg_type == 'inference_complete':
                    self.run_inference_btn.config(state='normal')
                    self.stop_inference_btn.config(state='disabled')
                    if self.inf_status_var.get() == "Inference complete!":
                        messagebox.showinfo(
                            "Complete", "Inference finished! Check the results summary below.")

                elif msg_type == 'val_progress':
                    self.val_progress_var.set(data)

                elif msg_type == 'val_status':
                    self.val_status_var.set(data)

                elif msg_type == 'val_results':
                    self.val_results_text.config(state='normal')
                    self.val_results_text.delete('1.0', tk.END)
                    self.val_results_text.insert('1.0', data)
                    self.val_results_text.config(state='disabled')

                elif msg_type == 'validation_test_complete':
                    self.run_val_btn.config(state='normal')
                    self.stop_val_btn.config(state='disabled')
                    if "complete" in self.val_status_var.get().lower():
                        messagebox.showinfo("Complete", "Test finished! Check results below.")

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self._process_messages)

    def run(self):
        """Start the GUI."""
        # Populate dataset dropdown once everything is built
        try:
            self._refresh_datasets()
        except Exception:
            pass
        self.root.mainloop()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    app = TrainingGUI()
    app.run()
