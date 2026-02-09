"""
Calibration GUI for Rho (ρ) Determination

A GUI application for calibrating the blur-to-depth conversion constant ρ.
Supports two modes:

- Estimated Optics (recommended): Enter rough guesses for optical params,
  fit ρ from data. The ρ value compensates for estimation errors.

- Known Optics: Enter exact camera specs. Fits small correction ρ ≈ 1.0.
  Use when you have documented/measured optical parameters.

Usage:
    python calibration_gui.py

Dependencies:
    pip install numpy opencv-python scipy pandas pyyaml matplotlib pillow
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
        import scipy
    except ImportError:
        missing.append("scipy")

    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    try:
        from PIL import Image, ImageTk
    except ImportError:
        missing.append("pillow")

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
        print("  pip install pyyaml numpy opencv-python scipy pandas pillow matplotlib")
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
import webbrowser
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import yaml
from PIL import Image, ImageTk

# Local imports
from blur_measurement import (
    measure_blur_auto, measure_blur_batch, detect_sphere, get_sphere_mask, BlurMeasurement
)
from calibration_core import (
    OpticalParams, CalibrationResultB, CalibrationResultHybrid,
    calibrate_approach_b, calibrate_hybrid,
    find_focal_plane, validate_calibration, export_calibration_yaml
)
from cine_loader import CineFolderLoader, check_pyphantom, PYPHANTOM_AVAILABLE

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not found. Install for visualizations: pip install matplotlib")


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class ZStackStats:
    """Statistics for loaded z-stack images."""
    num_images: int = 0
    image_width: int = 0
    image_height: int = 0
    z_min: float = 0.0
    z_max: float = 0.0
    z_step: float = 0.0
    focal_plane_idx: int = -1
    focal_plane_z: float = 0.0
    detected_sphere_center: Optional[Tuple[int, int]] = None
    detected_sphere_radius: Optional[int] = None


# =============================================================================
# Calibration GUI
# =============================================================================
class CalibrationGUI:
    """Main GUI application for calibration."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ρ Calibration Tool - Defocus Depth Estimation")
        self.root.geometry("1200x850")

        # Data storage
        self.zstack_images: List[np.ndarray] = []
        self.zstack_positions: List[float] = []
        self.zstack_filenames: List[str] = []
        self.zstack_stats: Optional[ZStackStats] = None
        self.cine_folder_loader: Optional[CineFolderLoader] = None  # For folder of .cine files

        self.blur_measurements: List[BlurMeasurement] = []
        self.sigma_values: List[float] = []

        self.calibration_b: Optional[CalibrationResultB] = None
        self.calibration_hybrid: Optional[CalibrationResultHybrid] = None

        # Multi-camera storage
        self.camera_calibrations: Dict[str, CalibrationResultHybrid] = {}
        self.focal_plane_offsets: Dict[str, float] = {}

        # Message queue for threading
        self.msg_queue = queue.Queue()

        # Preview image reference (prevent garbage collection)
        self._preview_photo = None

        # Build UI
        self._create_ui()

        # Start message processor
        self._process_messages()

    def _create_ui(self):
        """Create the main UI with 4 consolidated tabs."""
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill='both', expand=True)

        # Notebook with tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)

        # Create 4 consolidated tabs
        self.tab_data = ttk.Frame(self.notebook, padding=10)
        self.tab_calibrate = ttk.Frame(self.notebook, padding=10)
        self.tab_multicam = ttk.Frame(self.notebook, padding=10)
        self.tab_export = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_data, text="1. Data")
        self.notebook.add(self.tab_calibrate, text="2. Calibrate")
        self.notebook.add(self.tab_multicam, text="3. Multi-Camera")
        self.notebook.add(self.tab_export, text="4. Export")

        # Build tabs
        self._create_data_tab()
        self._create_calibrate_tab()
        self._create_multi_camera_tab()
        self._create_export_tab()

    # =========================================================================
    # Tab 1: Data (Load + Preview + Crop)
    # =========================================================================
    def _create_data_tab(self):
        """Create the Data tab (load, preview, auto-crop)."""
        # Two-column layout
        left_panel = ttk.Frame(self.tab_data)
        left_panel.pack(side='left', fill='both', expand=False, padx=(0, 10))

        right_panel = ttk.Frame(self.tab_data)
        right_panel.pack(side='left', fill='both', expand=True)

        # === LEFT PANEL ===

        # 1. Source type selection (TOP)
        source_type_frame = ttk.LabelFrame(left_panel, text="1. Z-Stack Source Type", padding=5)
        source_type_frame.pack(fill='x', pady=5)

        self.source_type_var = tk.StringVar(value="folder")
        ttk.Radiobutton(source_type_frame, text="Image Folder", variable=self.source_type_var,
                        value="folder", command=self._on_source_type_change).pack(side='left', padx=5)
        cine_rb = ttk.Radiobutton(source_type_frame, text=".cine File", variable=self.source_type_var,
                        value="cine", command=self._on_source_type_change)
        cine_rb.pack(side='left', padx=5)

        # Show pyphantom status
        pyphantom_available, pyphantom_msg = check_pyphantom()
        status_color = 'green' if pyphantom_available else 'orange'
        ttk.Label(source_type_frame, text=f"({'✓' if pyphantom_available else '⚠'} pyphantom)",
                  font=('', 8), foreground=status_color).pack(side='left', padx=10)

        # 2. Container for source-specific options (switches between folder/cine)
        self.source_container = ttk.Frame(left_panel)
        self.source_container.pack(fill='x', pady=0)

        # === Image Folder Source ===
        self.folder_source_frame = ttk.LabelFrame(self.source_container, text="Image Folder Source", padding=10)
        self.folder_source_frame.pack(fill='x', pady=5)

        row1 = ttk.Frame(self.folder_source_frame)
        row1.pack(fill='x', pady=2)
        ttk.Label(row1, text="Image Folder:", width=14).pack(side='left')
        self.zstack_folder_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self.zstack_folder_var, width=35).pack(side='left', padx=5)
        ttk.Button(row1, text="Browse", command=self._browse_zstack_folder).pack(side='left')

        # Positions file (optional)
        row2 = ttk.Frame(self.folder_source_frame)
        row2.pack(fill='x', pady=2)
        ttk.Label(row2, text="Positions CSV:", width=14).pack(side='left')
        self.positions_file_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.positions_file_var, width=35).pack(side='left', padx=5)
        ttk.Button(row2, text="Browse", command=self._browse_positions_file).pack(side='left')

        # CSV info feedback (blue text)
        self.positions_csv_info_var = tk.StringVar(value="")
        ttk.Label(
            self.folder_source_frame,
            textvariable=self.positions_csv_info_var,
            font=('TkDefaultFont', 8), foreground='blue'
        ).pack(anchor='w', pady=(2, 0))

        # Focus position (for converting stage positions to defocus)
        row3 = ttk.Frame(self.folder_source_frame)
        row3.pack(fill='x', pady=2)
        ttk.Label(row3, text="Focus at position:", width=14).pack(side='left')
        self.folder_focus_var = tk.StringVar(value="0")
        ttk.Entry(row3, textvariable=self.folder_focus_var, width=6).pack(side='left', padx=2)
        ttk.Label(row3, text="mm", font=('', 8)).pack(side='left', padx=2)
        ttk.Button(row3, text="Find Focus", command=self._find_focus_from_images).pack(side='left', padx=10)

        # Generate positions (inside folder source frame)
        gen_frame = ttk.Frame(self.folder_source_frame)
        gen_frame.pack(fill='x', pady=(10, 0))
        ttk.Label(gen_frame, text="Or generate positions:", font=('', 8, 'bold')).pack(anchor='w')

        range_row = ttk.Frame(gen_frame)
        range_row.pack(fill='x', pady=2)
        ttk.Label(range_row, text="Z min:", width=6).pack(side='left')
        self.z_min_var = tk.StringVar(value="-12")
        ttk.Entry(range_row, textvariable=self.z_min_var, width=6).pack(side='left', padx=(0, 5))
        ttk.Label(range_row, text="max:", width=4).pack(side='left')
        self.z_max_var = tk.StringVar(value="12")
        ttk.Entry(range_row, textvariable=self.z_max_var, width=6).pack(side='left', padx=(0, 5))
        ttk.Label(range_row, text="step:", width=4).pack(side='left')
        self.z_step_var = tk.StringVar(value="0.5")
        ttk.Entry(range_row, textvariable=self.z_step_var, width=6).pack(side='left', padx=(0, 5))
        ttk.Label(range_row, text="mm", font=('', 8)).pack(side='left')

        ttk.Button(gen_frame, text="Generate", command=self._generate_positions).pack(anchor='w', pady=2)
        self.positions_info_var = tk.StringVar(value="")
        ttk.Label(gen_frame, textvariable=self.positions_info_var, foreground='gray', font=('', 8)).pack(anchor='w')

        # === .cine Folder Source ===
        self.cine_source_frame = ttk.LabelFrame(self.source_container, text=".cine Folder Source", padding=10)
        # Not packed initially - shown when source type changes to "cine"

        cine_row1 = ttk.Frame(self.cine_source_frame)
        cine_row1.pack(fill='x', pady=2)
        ttk.Label(cine_row1, text=".cine Folder:", width=14).pack(side='left')
        self.cine_folder_var = tk.StringVar()
        ttk.Entry(cine_row1, textvariable=self.cine_folder_var, width=35).pack(side='left', padx=5)
        ttk.Button(cine_row1, text="Browse", command=self._browse_cine_folder).pack(side='left')

        # Positions CSV (optional - maps filenames to stage positions)
        cine_row1b = ttk.Frame(self.cine_source_frame)
        cine_row1b.pack(fill='x', pady=2)
        ttk.Label(cine_row1b, text="Positions CSV:", width=14).pack(side='left')
        self.cine_positions_csv_var = tk.StringVar()
        ttk.Entry(cine_row1b, textvariable=self.cine_positions_csv_var, width=35).pack(side='left', padx=5)
        ttk.Button(cine_row1b, text="Browse", command=self._browse_cine_positions_csv).pack(side='left')

        ttk.Label(self.cine_source_frame, text="(optional: CSV with filename, stage_position_mm columns)",
                  font=('', 8), foreground='gray').pack(anchor='w', padx=14)

        # Stage position mapping (fallback when no CSV)
        cine_row2 = ttk.Frame(self.cine_source_frame)
        cine_row2.pack(fill='x', pady=5)
        ttk.Label(cine_row2, text="Stage Range:", width=14).pack(side='left')
        ttk.Label(cine_row2, text="Start:", width=5).pack(side='left')
        self.stage_start_var = tk.StringVar(value="0")
        ttk.Entry(cine_row2, textvariable=self.stage_start_var, width=6).pack(side='left', padx=2)
        ttk.Label(cine_row2, text="End:", width=4).pack(side='left')
        self.stage_end_var = tk.StringVar(value="12")
        ttk.Entry(cine_row2, textvariable=self.stage_end_var, width=6).pack(side='left', padx=2)
        ttk.Label(cine_row2, text="mm (used if no CSV)", font=('', 8), foreground='gray').pack(side='left', padx=5)

        # Defocus offset (stage position where defocus = 0)
        cine_row3 = ttk.Frame(self.cine_source_frame)
        cine_row3.pack(fill='x', pady=2)
        ttk.Label(cine_row3, text="Focus at stage:", width=14).pack(side='left')
        self.stage_focus_var = tk.StringVar(value="6")
        ttk.Entry(cine_row3, textvariable=self.stage_focus_var, width=6).pack(side='left', padx=2)
        ttk.Label(cine_row3, text="mm", font=('', 8)).pack(side='left', padx=2)
        ttk.Button(cine_row3, text="Find Focus", command=self._find_focus_position).pack(side='left', padx=10)

        # Frame selection (which frame to extract from each .cine file)
        cine_row4 = ttk.Frame(self.cine_source_frame)
        cine_row4.pack(fill='x', pady=2)
        ttk.Label(cine_row4, text="Frame index:", width=14).pack(side='left')
        self.cine_frame_idx_var = tk.StringVar(value="0")
        ttk.Entry(cine_row4, textvariable=self.cine_frame_idx_var, width=6).pack(side='left', padx=2)
        ttk.Label(cine_row4, text="(which frame from each .cine)", font=('', 8), foreground='gray').pack(side='left', padx=5)

        self.cine_info_var = tk.StringVar(value="")
        ttk.Label(self.cine_source_frame, textvariable=self.cine_info_var, font=('', 8), foreground='blue').pack(anchor='w', pady=2)

        # Metadata
        meta_frame = ttk.LabelFrame(left_panel, text="Calibration Metadata", padding=10)
        meta_frame.pack(fill='x', pady=5)

        meta_row1 = ttk.Frame(meta_frame)
        meta_row1.pack(fill='x', pady=2)
        ttk.Label(meta_row1, text="Camera:", width=12).pack(side='left')
        self.camera_var = tk.StringVar(value="g")
        camera_combo = ttk.Combobox(meta_row1, textvariable=self.camera_var, values=["g", "m", "v", "custom"], width=12)
        camera_combo.pack(side='left')
        ttk.Label(meta_row1, text="(g=green, m=mono, v=violet)", font=('', 8), foreground='gray').pack(side='left', padx=5)

        meta_row2 = ttk.Frame(meta_frame)
        meta_row2.pack(fill='x', pady=2)
        ttk.Label(meta_row2, text="Aperture:", width=12).pack(side='left')
        self.aperture_var = tk.StringVar(value="position_1")
        ttk.Entry(meta_row2, textvariable=self.aperture_var, width=15).pack(side='left')
        ttk.Label(meta_row2, text="(f/4, f/5.6, or custom label)", font=('', 8), foreground='gray').pack(side='left', padx=5)

        # Load button
        load_btn_frame = ttk.Frame(left_panel)
        load_btn_frame.pack(fill='x', pady=10)

        self.load_btn = ttk.Button(load_btn_frame, text="Load Z-Stack Images", command=self._load_zstack)
        self.load_btn.pack(side='left', padx=5)

        # Auto-crop section
        crop_frame = ttk.LabelFrame(left_panel, text="Auto-Crop to Sphere", padding=5)
        crop_frame.pack(fill='x', pady=5)

        crop_row1 = ttk.Frame(crop_frame)
        crop_row1.pack(fill='x', pady=2)
        ttk.Label(crop_row1, text="Padding (px):", width=12).pack(side='left')
        self.crop_padding_var = tk.StringVar(value="50")
        ttk.Entry(crop_row1, textvariable=self.crop_padding_var, width=8).pack(side='left')
        ttk.Label(crop_row1, text="around sphere", font=('', 8), foreground='gray').pack(side='left', padx=5)

        crop_row2 = ttk.Frame(crop_frame)
        crop_row2.pack(fill='x', pady=2)
        self.crop_btn = ttk.Button(crop_row2, text="Auto-Crop All", command=self._auto_crop_to_sphere)
        self.crop_btn.pack(side='left', padx=2)
        ttk.Button(crop_row2, text="Save Cropped", command=self._save_cropped_images).pack(side='left', padx=2)
        self.crop_status_var = tk.StringVar(value="")
        ttk.Label(crop_row2, textvariable=self.crop_status_var, font=('', 8), foreground='blue').pack(side='left', padx=5)

        # Loading progress
        self.load_progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(left_panel, variable=self.load_progress_var, maximum=100).pack(fill='x', pady=5)

        # Status
        self.load_status_var = tk.StringVar(value="No images loaded")
        ttk.Label(left_panel, textvariable=self.load_status_var).pack(anchor='w')

        # Statistics display
        stats_frame = ttk.LabelFrame(left_panel, text="Z-Stack Statistics", padding=10)
        stats_frame.pack(fill='x', pady=5)

        self.zstack_stats_text = tk.Text(stats_frame, height=8, width=45, state='disabled', font=('Courier', 9))
        self.zstack_stats_text.pack(fill='x')

        # === RIGHT PANEL ===

        # Preview
        preview_frame = ttk.LabelFrame(right_panel, text="Image Preview", padding=10)
        preview_frame.pack(fill='both', expand=True)

        # Preview canvas with scrollbars
        canvas_container = ttk.Frame(preview_frame)
        canvas_container.pack(fill='both', expand=True)

        # Get the ttk theme background color
        style = ttk.Style()
        bg_color = style.lookup('TFrame', 'background') or self.root.cget('bg')

        self.preview_canvas = tk.Canvas(canvas_container, width=450, height=450, bg=bg_color, highlightthickness=0, bd=0)
        self.preview_canvas.pack(pady=5)

        # Position slider
        slider_frame = ttk.Frame(preview_frame)
        slider_frame.pack(fill='x', pady=5)

        ttk.Label(slider_frame, text="Position:").pack(side='left')
        self.preview_slider = ttk.Scale(
            slider_frame, from_=0, to=100, orient='horizontal',
            command=self._on_preview_slider
        )
        self.preview_slider.pack(fill='x', side='left', expand=True, padx=10)

        self.preview_label_var = tk.StringVar(value="z = 0.0 mm")
        ttk.Label(slider_frame, textvariable=self.preview_label_var, width=15).pack(side='left')

        # Sphere detection and utilities
        focal_frame = ttk.Frame(preview_frame)
        focal_frame.pack(fill='x', pady=5)

        ttk.Button(focal_frame, text="Detect Sphere", command=self._detect_sphere_in_preview).pack(side='left', padx=5)
        ttk.Label(focal_frame, text="|", foreground='gray').pack(side='left', padx=5)
        ttk.Button(focal_frame, text="Verify", command=self._auto_detect_focal, width=6).pack(side='left')
        ttk.Label(focal_frame, text="(verify sharpest is at z=0)", foreground='gray', font=('', 8)).pack(side='left', padx=5)

        self.focal_info_var = tk.StringVar(value="")
        ttk.Label(focal_frame, textvariable=self.focal_info_var, foreground='blue', font=('', 9)).pack(side='left', padx=10)

    # =========================================================================
    # Tab 2: Calibrate (Measure + Fit ρ combined)
    # =========================================================================
    def _create_calibrate_tab(self):
        """Create the Calibrate tab (measure blur + fit ρ)."""
        # Three-column layout: Measure | Fit | Results
        columns_frame = ttk.Frame(self.tab_calibrate)
        columns_frame.pack(fill='both', expand=True)

        # Column 1: Measurement settings
        measure_col = ttk.Frame(columns_frame, width=280)
        measure_col.pack(side='left', fill='y', padx=(0, 5))
        measure_col.pack_propagate(False)

        # Column 2: Calibration settings
        calib_col = ttk.Frame(columns_frame, width=280)
        calib_col.pack(side='left', fill='y', padx=5)
        calib_col.pack_propagate(False)

        # Column 3: Results (expandable)
        results_col = ttk.Frame(columns_frame)
        results_col.pack(side='left', fill='both', expand=True, padx=(5, 0))

        # === COLUMN 1: MEASUREMENT ===
        measure_header = ttk.Label(measure_col, text="Step 1: Measure Blur", font=('TkDefaultFont', 10, 'bold'))
        measure_header.pack(anchor='w', pady=(0, 5))

        # Blur measurement method (sigmoid edge fitting)
        self.blur_method_var = tk.StringVar(value="sigmoid")

        # Sphere detection
        sphere_frame = ttk.LabelFrame(measure_col, text="Sphere Detection", padding=5)
        sphere_frame.pack(fill='x', pady=2)

        self.auto_detect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sphere_frame, text="Auto-detect", variable=self.auto_detect_var, command=self._on_auto_detect_toggle).pack(anchor='w')

        coord_row = ttk.Frame(sphere_frame)
        coord_row.pack(fill='x', pady=2)
        ttk.Label(coord_row, text="X:", width=3).pack(side='left')
        self.sphere_cx_var = tk.StringVar(value="")
        self.sphere_cx_entry = ttk.Entry(coord_row, textvariable=self.sphere_cx_var, width=6)
        self.sphere_cx_entry.pack(side='left', padx=(0, 5))
        ttk.Label(coord_row, text="Y:", width=3).pack(side='left')
        self.sphere_cy_var = tk.StringVar(value="")
        self.sphere_cy_entry = ttk.Entry(coord_row, textvariable=self.sphere_cy_var, width=6)
        self.sphere_cy_entry.pack(side='left', padx=(0, 5))
        ttk.Label(coord_row, text="R:", width=3).pack(side='left')
        self.sphere_r_var = tk.StringVar(value="")
        self.sphere_r_entry = ttk.Entry(coord_row, textvariable=self.sphere_r_var, width=6)
        self.sphere_r_entry.pack(side='left')

        self._on_auto_detect_toggle()

        # Measure button
        self.measure_btn = ttk.Button(measure_col, text="▶ Measure All", command=self._measure_blur)
        self.measure_btn.pack(fill='x', pady=5)

        # Progress
        self.measure_progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(measure_col, variable=self.measure_progress_var, maximum=100).pack(fill='x', pady=2)
        self.measure_status_var = tk.StringVar(value="Ready")
        ttk.Label(measure_col, textvariable=self.measure_status_var, font=('', 8)).pack(anchor='w')

        # Summary
        summary_frame = ttk.LabelFrame(measure_col, text="Summary", padding=5)
        summary_frame.pack(fill='x', pady=5)
        self.measure_summary_text = tk.Text(summary_frame, height=5, width=30, state='disabled', font=('Courier', 8))
        self.measure_summary_text.pack(fill='x')

        # === COLUMN 2: CALIBRATION ===
        calib_header = ttk.Label(calib_col, text="Step 2: Fit ρ", font=('TkDefaultFont', 10, 'bold'))
        calib_header.pack(anchor='w', pady=(0, 5))

        # Calibration Mode selection (NEW - before approach)
        cal_mode_frame = ttk.LabelFrame(calib_col, text="Calibration Mode", padding=5)
        cal_mode_frame.pack(fill='x', pady=2)

        self.calibration_mode_var = tk.StringVar(value="optical")

        mode_row = ttk.Frame(cal_mode_frame)
        mode_row.pack(fill='x', pady=2)

        ttk.Radiobutton(
            mode_row,
            text="Optical Formula",
            variable=self.calibration_mode_var,
            value="optical",
            command=self._on_calibration_mode_change
        ).pack(side='left', padx=10)

        ttk.Radiobutton(
            mode_row,
            text="Direct Calibration",
            variable=self.calibration_mode_var,
            value="direct",
            command=self._on_calibration_mode_change
        ).pack(side='left', padx=10)

        # Mode description
        cal_mode_desc = ttk.Label(
            cal_mode_frame,
            text="Select mode to show relevant calibration parameters.",
            font=('', 8),
            foreground='gray'
        )
        cal_mode_desc.pack(anchor='w', pady=(5, 0))

        # Internal approach state (was previously a UI selector, now always "hybrid" = recommended)
        self.approach_var = tk.StringVar(value="hybrid")

        # Optical parameters
        self.optical_frame = ttk.LabelFrame(calib_col, text="Optical Params", padding=5)
        self.optical_frame.pack(fill='x', pady=2)

        params = [("f (mm):", "focal_length", "50.0"), ("N:", "f_number", "4.0"),
                  ("D (mm):", "focus_distance", "300.0"), ("px (mm):", "pixel_size", "0.01")]

        self.optical_vars = {}
        for label, key, default in params:
            row = ttk.Frame(self.optical_frame)
            row.pack(fill='x', pady=1)
            ttk.Label(row, text=label, width=8).pack(side='left')
            var = tk.StringVar(value=default)
            self.optical_vars[key] = var
            ttk.Entry(row, textvariable=var, width=10).pack(side='left')
            # Add [?] info button for pixel_size to help lookup camera datasheet
            if key == 'pixel_size':
                ttk.Button(row, text="?", width=2, command=self._lookup_pixel_size).pack(side='left', padx=2)

        # Reference defocus (only for Estimated Optics)
        self.ref_row = ttk.Frame(self.optical_frame)
        self.ref_row.pack(fill='x', pady=1)
        ttk.Label(self.ref_row, text="Ref d:", width=8).pack(side='left')
        self.ref_defocus_var = tk.StringVar(value="5.0")
        ttk.Entry(self.ref_row, textvariable=self.ref_defocus_var, width=10).pack(side='left')
        ttk.Label(self.ref_row, text="mm", font=('', 7)).pack(side='left')

        # Calibrate button
        self.calibrate_btn = ttk.Button(calib_col, text="▶ Calibrate ρ", command=self._run_calibration)
        self.calibrate_btn.pack(fill='x', pady=5)

        # Fit quality summary
        quality_frame = ttk.LabelFrame(calib_col, text="Fit Quality", padding=5)
        quality_frame.pack(fill='x', pady=2)
        self.fit_quality_text = tk.Text(quality_frame, height=4, width=30, state='disabled', font=('Courier', 8))
        self.fit_quality_text.pack(fill='x')

        # === COLUMN 3: RESULTS ===
        results_header = ttk.Label(results_col, text="Results", font=('TkDefaultFont', 10, 'bold'))
        results_header.pack(anchor='w', pady=(0, 5))

        # Results text
        self.results_text = scrolledtext.ScrolledText(results_col, height=10, state='disabled', font=('Courier', 9))
        self.results_text.pack(fill='x', pady=2)

        # Plot
        if HAS_MATPLOTLIB:
            plot_frame = ttk.LabelFrame(results_col, text="Calibration Curve", padding=5)
            plot_frame.pack(fill='both', expand=True, pady=2)

            self.calib_fig = Figure(figsize=(5, 3), dpi=100)
            self.calib_ax = self.calib_fig.add_subplot(111)
            self.calib_canvas = FigureCanvasTkAgg(self.calib_fig, plot_frame)
            self.calib_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Measurements table (compact)
        table_frame = ttk.LabelFrame(results_col, text="Measurements", padding=2)
        table_frame.pack(fill='x', pady=2)

        columns = ('z_mm', 'sigma_px', 'conf')
        self.measure_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=5)
        self.measure_tree.heading('z_mm', text='Z (mm)')
        self.measure_tree.heading('sigma_px', text='σ (px)')
        self.measure_tree.heading('conf', text='Conf')
        self.measure_tree.column('z_mm', width=70)
        self.measure_tree.column('sigma_px', width=70)
        self.measure_tree.column('conf', width=50)

        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.measure_tree.yview)
        self.measure_tree.configure(yscrollcommand=scrollbar.set)
        self.measure_tree.pack(side='left', fill='x', expand=True)
        scrollbar.pack(side='right', fill='y')

    # =========================================================================
    # Tab 3: Multi-Camera (Sign Resolution)
    # =========================================================================
    def _create_multi_camera_tab(self):
        """Create the Multi-Camera tab for sign resolution setup."""
        # Single column layout (simpler now)
        main_frame = ttk.Frame(self.tab_multicam)
        main_frame.pack(fill='both', expand=True)

        # Header
        header = ttk.Label(main_frame, text="Multi-Camera Sign Resolution", font=('TkDefaultFont', 11, 'bold'))
        header.pack(anchor='w', pady=(0, 10))

        info_text = "With two cameras at different focal planes, compare blur to determine depth sign (front vs behind)."
        ttk.Label(main_frame, text=info_text, foreground='gray', wraplength=500).pack(anchor='w', pady=(0, 10))

        # Two-column content
        content = ttk.Frame(main_frame)
        content.pack(fill='both', expand=True)

        left_col = ttk.Frame(content)
        left_col.pack(side='left', fill='both', expand=True, padx=(0, 10))

        right_col = ttk.Frame(content)
        right_col.pack(side='left', fill='both', expand=True)

        # === LEFT: Camera Calibrations ===
        cam_frame = ttk.LabelFrame(left_col, text="Camera Calibrations", padding=10)
        cam_frame.pack(fill='both', expand=True)

        columns = ('camera', 'rho', 'focal_offset')
        self.camera_tree = ttk.Treeview(cam_frame, columns=columns, show='headings', height=6)
        self.camera_tree.heading('camera', text='Camera')
        self.camera_tree.heading('rho', text='ρ (px/mm)')
        self.camera_tree.heading('focal_offset', text='Focal Offset (mm)')
        self.camera_tree.column('camera', width=80)
        self.camera_tree.column('rho', width=100)
        self.camera_tree.column('focal_offset', width=120)
        self.camera_tree.pack(fill='both', expand=True)

        # Add controls
        add_frame = ttk.Frame(cam_frame)
        add_frame.pack(fill='x', pady=(10, 0))

        ttk.Label(add_frame, text="Focal offset from reference (mm):").pack(side='left')
        self.focal_offset_var = tk.StringVar(value="0.0")
        ttk.Entry(add_frame, textvariable=self.focal_offset_var, width=10).pack(side='left', padx=5)

        btn_frame = ttk.Frame(cam_frame)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="Add Current Calibration", command=self._add_camera_calibration).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Remove Selected", command=self._remove_camera_calibration).pack(side='left', padx=2)

        ttk.Label(cam_frame, text="First camera added becomes reference (offset = 0)",
                  font=('', 8), foreground='gray').pack(anchor='w')

        # === RIGHT: Test Sign Resolution ===
        test_frame = ttk.LabelFrame(right_col, text="Test Sign Resolution", padding=10)
        test_frame.pack(fill='x')

        ttk.Label(test_frame, text="Enter blur from two cameras for same droplet:").pack(anchor='w', pady=(0, 10))

        row1 = ttk.Frame(test_frame)
        row1.pack(fill='x', pady=2)
        ttk.Label(row1, text="Camera 1 σ (px):", width=15).pack(side='left')
        self.test_sigma1_var = tk.StringVar(value="18.0")
        ttk.Entry(row1, textvariable=self.test_sigma1_var, width=10).pack(side='left')

        row2 = ttk.Frame(test_frame)
        row2.pack(fill='x', pady=2)
        ttk.Label(row2, text="Camera 2 σ (px):", width=15).pack(side='left')
        self.test_sigma2_var = tk.StringVar(value="7.2")
        ttk.Entry(row2, textvariable=self.test_sigma2_var, width=10).pack(side='left')

        ttk.Button(test_frame, text="Calculate Signed Depth", command=self._test_sign_resolution).pack(pady=10)

        # Result
        self.sign_result_var = tk.StringVar(value="")
        ttk.Label(test_frame, textvariable=self.sign_result_var, font=('TkDefaultFont', 14, 'bold')).pack()

        self.sign_explanation_var = tk.StringVar(value="")
        ttk.Label(test_frame, textvariable=self.sign_explanation_var, foreground='gray', wraplength=300).pack(pady=5)

        # Example
        example_frame = ttk.LabelFrame(right_col, text="How It Works", padding=10)
        example_frame.pack(fill='x', pady=10)

        example_text = """Camera g: focal plane at z = 0 mm
Camera m: focal plane at z = +3 mm (behind g)

Droplet at z = +5 mm:
  • g sees 5mm offset → more blur (σ = 18px)
  • m sees 2mm offset → less blur (σ = 7px)
  • m is sharper → droplet is BEHIND g's focal plane
  • Signed depth = +5 mm"""
        ttk.Label(example_frame, text=example_text, justify='left', font=('Courier', 9)).pack(anchor='w')

    # =========================================================================
    # Tab 4: Export
    # =========================================================================
    def _create_export_tab(self):
        """Create the Export tab."""
        # Two-column layout
        left_col = ttk.Frame(self.tab_export)
        left_col.pack(side='left', fill='both', expand=False, padx=(0, 10))

        right_col = ttk.Frame(self.tab_export)
        right_col.pack(side='left', fill='both', expand=True)

        # === LEFT COLUMN ===

        # Export options
        options_frame = ttk.LabelFrame(left_col, text="Export Options", padding=10)
        options_frame.pack(fill='x', pady=5)

        self.export_yaml_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="YAML config (for Training GUI)", variable=self.export_yaml_var).pack(anchor='w')
        ttk.Label(options_frame, text="    calibration_results.yaml", font=('', 8), foreground='gray').pack(anchor='w')

        self.export_csv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="CSV measurements", variable=self.export_csv_var).pack(anchor='w', pady=(5, 0))
        ttk.Label(options_frame, text="    measurements.csv (z, σ pairs)", font=('', 8), foreground='gray').pack(anchor='w')

        self.export_plots_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Plots (PNG)", variable=self.export_plots_var).pack(anchor='w', pady=(5, 0))
        ttk.Label(options_frame, text="    calibration_curve.png", font=('', 8), foreground='gray').pack(anchor='w')

        # Output folder
        folder_frame = ttk.LabelFrame(left_col, text="Output Location", padding=10)
        folder_frame.pack(fill='x', pady=5)

        folder_row = ttk.Frame(folder_frame)
        folder_row.pack(fill='x', pady=2)
        self.export_folder_var = tk.StringVar(value=str(Path(__file__).parent / "calibration_output"))
        ttk.Entry(folder_row, textvariable=self.export_folder_var, width=40).pack(side='left', padx=(0, 5))
        ttk.Button(folder_row, text="Browse", command=self._browse_export_folder).pack(side='left')

        # Export button
        export_btn_frame = ttk.Frame(left_col)
        export_btn_frame.pack(fill='x', pady=10)

        self.export_btn = ttk.Button(export_btn_frame, text="Export Calibration", command=self._export_calibration)
        self.export_btn.pack(side='left', padx=5)

        ttk.Button(export_btn_frame, text="Copy ρ to Clipboard", command=self._copy_rho).pack(side='left')

        self.export_status_var = tk.StringVar(value="")
        ttk.Label(left_col, textvariable=self.export_status_var, foreground='green').pack(anchor='w', pady=5)

        # Next steps
        next_frame = ttk.LabelFrame(left_col, text="Next Steps", padding=10)
        next_frame.pack(fill='x', pady=5)

        next_text = """After calibration:
1. Open the Training GUI
2. Load your sharp crops
3. Enter the optical parameters
4. Enter ρ = formula_rho value
5. Generate synthetic training data
6. Train your model

The synthetic blur will match your camera!"""

        ttk.Label(next_frame, text=next_text, justify='left', font=('TkDefaultFont', 9)).pack(anchor='w')

        # === RIGHT COLUMN ===

        # YAML preview
        preview_frame = ttk.LabelFrame(right_col, text="YAML Preview", padding=10)
        preview_frame.pack(fill='both', expand=True, pady=5)

        self.export_preview = scrolledtext.ScrolledText(preview_frame, height=25, state='disabled', font=('Courier', 9))
        self.export_preview.pack(fill='both', expand=True)

        ttk.Button(preview_frame, text="Refresh Preview", command=self._update_export_preview).pack(pady=5)

    # =========================================================================
    # Event Handlers - Tab 1
    # =========================================================================
    def _on_source_type_change(self):
        """Handle source type radio button change."""
        source_type = self.source_type_var.get()

        if source_type == "folder":
            # Show folder source, hide cine source
            self.cine_source_frame.pack_forget()
            self.folder_source_frame.pack(in_=self.source_container, fill='x', pady=5)
        else:
            # Show cine source, hide folder source
            self.folder_source_frame.pack_forget()
            self.cine_source_frame.pack(in_=self.source_container, fill='x', pady=5)

            if not PYPHANTOM_AVAILABLE:
                messagebox.showwarning("pyphantom Not Available",
                    "The pyphantom library is not installed.\n\n"
                    "To load .cine files, install the Phantom SDK and pyphantom package.\n\n"
                    "Alternatively, export your z-stack as individual images.")

    def _browse_cine_folder(self):
        """Browse for folder containing .cine files."""
        folder = filedialog.askdirectory(title="Select Folder with .cine Files")
        if folder:
            self.cine_folder_var.set(folder)
            # Try to scan and show info
            self._preview_cine_folder_info()

    def _preview_cine_folder_info(self):
        """Preview info about selected .cine folder."""
        folder_path = self.cine_folder_var.get()
        if not folder_path:
            return

        loader = CineFolderLoader(folder_path)
        if loader.num_files > 0:
            info = loader.get_info()
            self.cine_info_var.set(
                f"Found: {info['num_files']} .cine files, {info['image_width']}x{info['image_height']} px"
            )
            self.cine_folder_loader = loader
        else:
            self.cine_info_var.set("No .cine files found in folder")
            self.cine_folder_loader = None

    def _browse_cine_positions_csv(self):
        """Browse for CSV file mapping .cine filenames to stage positions."""
        file = filedialog.askopenfilename(
            title="Select Positions CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file:
            self.cine_positions_csv_var.set(file)
            # Preview the CSV contents
            self._preview_cine_positions_csv(file)

    def _preview_cine_positions_csv(self, csv_path: str):
        """Preview contents of positions CSV."""
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)

            if 'filename' not in df.columns:
                self.cine_info_var.set("CSV missing 'filename' column")
                return

            pos_col = None
            if 'stage_position_mm' in df.columns:
                pos_col = 'stage_position_mm'
            elif 'z_position_mm' in df.columns:
                pos_col = 'z_position_mm'
            elif 'position' in df.columns:
                pos_col = 'position'

            if pos_col is None:
                self.cine_info_var.set("CSV missing position column (stage_position_mm or z_position_mm)")
                return

            positions = df[pos_col].values
            self.cine_info_var.set(
                f"CSV: {len(df)} entries, positions {positions.min():.1f} to {positions.max():.1f} mm"
            )

            # Auto-fill stage range from CSV
            self.stage_start_var.set(f"{positions.min():.1f}")
            self.stage_end_var.set(f"{positions.max():.1f}")

        except Exception as e:
            self.cine_info_var.set(f"Error reading CSV: {e}")

    def _find_focus_position(self):
        """Find focus position by measuring sharpness across z-stack."""
        folder_path = self.cine_folder_var.get()
        if not folder_path:
            messagebox.showerror("Error", "Please select a .cine folder first")
            return

        if not PYPHANTOM_AVAILABLE:
            messagebox.showerror("Error", "pyphantom not available")
            return

        # Ensure folder is loaded
        if self.cine_folder_loader is None or str(self.cine_folder_loader.folder) != folder_path:
            self.cine_folder_loader = CineFolderLoader(folder_path)
            if self.cine_folder_loader.num_files == 0:
                messagebox.showerror("Error", "No .cine files found in folder")
                return

        # Get stage range
        try:
            stage_start = float(self.stage_start_var.get())
            stage_end = float(self.stage_end_var.get())
            frame_idx = int(self.cine_frame_idx_var.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameters: {e}")
            return

        self.load_status_var.set("Measuring sharpness to find focus...")
        self.root.update_idletasks()

        # Load frames and measure sharpness
        def progress_callback(current, total):
            self.load_progress_var.set(current / total * 100)
            self.root.update_idletasks()

        # Use temporary defocus = 0 (we'll calculate stage positions)
        images, _, filenames = self.cine_folder_loader.load_zstack(
            z_start=stage_start,
            z_end=stage_end,
            frame_idx=frame_idx,
            progress_callback=progress_callback
        )

        if not images:
            messagebox.showerror("Error", "No frames extracted")
            return

        # Measure sharpness (Laplacian variance) for each frame
        sharpness_values = []
        for img in images:
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_values.append(sharpness)

        # Find the sharpest frame
        max_idx = np.argmax(sharpness_values)

        # Calculate the corresponding stage position
        n_files = len(images)
        if n_files == 1:
            focus_stage = (stage_start + stage_end) / 2
        else:
            t = max_idx / (n_files - 1)
            focus_stage = stage_start + t * (stage_end - stage_start)

        # Update the focus position
        self.stage_focus_var.set(f"{focus_stage:.2f}")

        # Show results
        self._update_stats_text(f"Focus Detection Results\n" + "=" * 40)
        self._append_stats_text(f"\n\nAnalyzed {len(images)} frames")
        self._append_stats_text(f"\nSharpest frame: #{max_idx + 1} of {n_files}")
        self._append_stats_text(f"\nFilename: {filenames[max_idx]}")
        self._append_stats_text(f"\nSharpness (Laplacian var): {sharpness_values[max_idx]:.1f}")
        self._append_stats_text(f"\n\nEstimated focus at stage: {focus_stage:.2f} mm")
        self._append_stats_text(f"\n\n(Focus position has been auto-filled)")

        # Show sharpness curve summary
        self._append_stats_text(f"\n\nSharpness range: {min(sharpness_values):.1f} to {max(sharpness_values):.1f}")

        self.load_status_var.set(f"Focus found at stage {focus_stage:.2f} mm")
        self.load_progress_var.set(100)

    def _browse_zstack_folder(self):
        folder = filedialog.askdirectory(title="Select Z-Stack Image Folder")
        if folder:
            self.zstack_folder_var.set(folder)

    def _browse_positions_file(self):
        file = filedialog.askopenfilename(
            title="Select Positions CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file:
            self.positions_file_var.set(file)
            self._preview_positions_csv(file)

    def _preview_positions_csv(self, csv_path: str):
        """Preview contents of positions CSV and show info."""
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)

            if len(df.columns) < 2:
                self.positions_csv_info_var.set("CSV needs at least 2 columns")
                return

            # Use second column as positions (first is filename)
            positions = df.iloc[:, 1].astype(float).values
            n_positions = len(positions)
            pos_min = positions.min()
            pos_max = positions.max()

            # Calculate approximate step
            if n_positions > 1:
                sorted_pos = np.sort(positions)
                steps = np.diff(sorted_pos)
                avg_step = np.mean(steps)
                self.positions_csv_info_var.set(
                    f"CSV: {n_positions} positions, {pos_min:.1f} to {pos_max:.1f} mm, step ~{avg_step:.2f} mm"
                )
            else:
                self.positions_csv_info_var.set(f"CSV: {n_positions} position at {pos_min:.1f} mm")

        except Exception as e:
            self.positions_csv_info_var.set(f"Error reading CSV: {e}")

    def _find_focus_from_images(self):
        """Find focus position by measuring sharpness across images."""
        folder = self.zstack_folder_var.get()
        if not folder:
            messagebox.showerror("Error", "Please select an image folder first")
            return

        folder = Path(folder)
        if not folder.exists():
            messagebox.showerror("Error", f"Folder not found: {folder}")
            return

        # Find images
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))
        image_files = sorted(set(image_files))

        if not image_files:
            messagebox.showerror("Error", "No images found in folder")
            return

        # Load positions from CSV if available
        positions_file = self.positions_file_var.get()
        positions = []

        if positions_file and Path(positions_file).exists():
            try:
                import pandas as pd
                df = pd.read_csv(positions_file)
                # Build position dict (first col = filename, second col = position)
                filenames_col = df.iloc[:, 0].astype(str)
                positions_col = df.iloc[:, 1].astype(float)
                pos_dict = {}
                for fn, pos in zip(filenames_col, positions_col):
                    pos_dict[fn] = pos
                    pos_dict[Path(fn).stem] = pos

                # Match positions to images
                for img_path in image_files:
                    if img_path.name in pos_dict:
                        positions.append(pos_dict[img_path.name])
                    elif img_path.stem in pos_dict:
                        positions.append(pos_dict[img_path.stem])
                    else:
                        positions.append(None)
            except Exception as e:
                messagebox.showerror("Error", f"Error reading CSV: {e}")
                return
        else:
            # No CSV - use generated positions or linear interpolation
            try:
                z_min = float(self.z_min_var.get())
                z_max = float(self.z_max_var.get())
                n_files = len(image_files)
                for i in range(n_files):
                    z = z_min + (z_max - z_min) * i / (n_files - 1) if n_files > 1 else z_min
                    positions.append(z)
            except ValueError:
                messagebox.showerror("Error", "Invalid Z range values")
                return

        self.load_status_var.set("Measuring sharpness to find focus...")
        self.root.update_idletasks()

        # Measure sharpness for each image
        sharpness_values = []
        valid_positions = []
        valid_filenames = []

        for i, img_path in enumerate(image_files):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None and positions[i] is not None:
                laplacian = cv2.Laplacian(img, cv2.CV_64F)
                sharpness = laplacian.var()
                sharpness_values.append(sharpness)
                valid_positions.append(positions[i])
                valid_filenames.append(img_path.name)

            self.load_progress_var.set((i + 1) / len(image_files) * 100)
            self.root.update_idletasks()

        if not sharpness_values:
            messagebox.showerror("Error", "Could not measure sharpness")
            return

        # Find the sharpest frame
        max_idx = np.argmax(sharpness_values)
        focus_position = valid_positions[max_idx]

        # Update the focus position
        self.folder_focus_var.set(f"{focus_position:.2f}")

        # Show results
        self._update_stats_text(f"Focus Detection Results\n" + "=" * 40)
        self._append_stats_text(f"\n\nAnalyzed {len(sharpness_values)} images")
        self._append_stats_text(f"\nSharpest image: #{max_idx + 1}")
        self._append_stats_text(f"\nFilename: {valid_filenames[max_idx]}")
        self._append_stats_text(f"\nPosition: {focus_position:.2f} mm")
        self._append_stats_text(f"\nSharpness (Laplacian var): {sharpness_values[max_idx]:.1f}")
        self._append_stats_text(f"\n\nFocus position has been auto-filled.")
        self._append_stats_text(f"\nDefocus will be calculated as: position - {focus_position:.2f}")

        self.load_status_var.set(f"Focus found at position {focus_position:.2f} mm")
        self.load_progress_var.set(100)

    def _generate_positions(self):
        """Generate positions from range."""
        try:
            z_min = float(self.z_min_var.get())
            z_max = float(self.z_max_var.get())
            z_step = float(self.z_step_var.get())

            positions = list(np.arange(z_min, z_max + z_step / 2, z_step))
            self.zstack_positions = positions

            self.positions_info_var.set(f"Generated {len(positions)} positions: {z_min} to {z_max} mm, step {z_step}")

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid range values: {e}")

    def _load_zstack(self):
        """Load z-stack images from folder or .cine file."""
        source_type = self.source_type_var.get()

        if source_type == "cine":
            self._load_zstack_from_cine()
        else:
            self._load_zstack_from_folder()

    def _load_zstack_from_cine(self):
        """Load z-stack from folder of .cine files."""
        folder_path = self.cine_folder_var.get()
        if not folder_path:
            messagebox.showerror("Error", "Please select a .cine folder")
            return

        if not Path(folder_path).exists():
            messagebox.showerror("Error", f"Folder not found: {folder_path}")
            return

        if not PYPHANTOM_AVAILABLE:
            messagebox.showerror("Error", "pyphantom not available - cannot load .cine files")
            return

        # Load folder if not already loaded
        if self.cine_folder_loader is None or str(self.cine_folder_loader.folder) != folder_path:
            self.cine_folder_loader = CineFolderLoader(folder_path)
            if self.cine_folder_loader.num_files == 0:
                messagebox.showerror("Error", "No .cine files found in folder")
                return

        # Get stage range and focus offset
        try:
            stage_start = float(self.stage_start_var.get())
            stage_end = float(self.stage_end_var.get())
            stage_focus = float(self.stage_focus_var.get())
            frame_idx = int(self.cine_frame_idx_var.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameters: {e}")
            return

        self.load_status_var.set("Extracting frames from .cine files...")
        self.root.update_idletasks()

        # Extract one frame from each .cine file
        def progress_callback(current, total):
            self.load_progress_var.set(current / total * 100)
            self.root.update_idletasks()

        # Check if CSV is provided for position mapping
        csv_path = self.cine_positions_csv_var.get()
        use_csv = csv_path and Path(csv_path).exists()

        if use_csv:
            # Load using CSV for position mapping
            images, positions, filenames = self.cine_folder_loader.load_with_positions_csv(
                csv_path=csv_path,
                stage_offset=stage_focus,
                frame_idx=frame_idx
            )
            position_source = "CSV"
        else:
            # Use linear interpolation across stage range
            z_start = stage_start - stage_focus
            z_end = stage_end - stage_focus

            images, positions, filenames = self.cine_folder_loader.load_zstack(
                z_start=z_start,
                z_end=z_end,
                frame_idx=frame_idx,
                progress_callback=progress_callback
            )
            position_source = "linear interpolation"

        if not images:
            messagebox.showerror("Error", "No frames extracted from .cine files")
            return

        self.zstack_images = images
        self.zstack_positions = positions
        self.zstack_filenames = filenames

        # Create stats
        h, w = self.zstack_images[0].shape
        z_min = min(self.zstack_positions)
        z_max = max(self.zstack_positions)
        n_pos = len(self.zstack_positions)
        z_step = (z_max - z_min) / (n_pos - 1) if n_pos > 1 else 0

        self.zstack_stats = ZStackStats(
            num_images=len(self.zstack_images),
            image_width=w,
            image_height=h,
            z_min=z_min,
            z_max=z_max,
            z_step=z_step
        )

        # Update slider
        self.preview_slider.configure(to=len(self.zstack_images) - 1)
        self._update_preview(0)

        # Update stats display
        self._update_stats_text(f"Loaded Z-Stack from .cine folder\n" + "=" * 40)
        self._append_stats_text(f"\nSource: {Path(folder_path).name}")
        self._append_stats_text(f"\n.cine files: {self.zstack_stats.num_images}")
        self._append_stats_text(f"\nSize: {w} × {h} px")
        self._append_stats_text(f"\nPositions from: {position_source}")
        self._append_stats_text(f"\nFocus at stage: {stage_focus} mm")
        self._append_stats_text(f"\nDefocus range: {self.zstack_stats.z_min:.1f} to {self.zstack_stats.z_max:.1f} mm")
        self._append_stats_text(f"\nZ step: {z_step:.3f} mm")

        self.load_status_var.set(f"Loaded {len(self.zstack_images)} frames from .cine folder")
        self.load_progress_var.set(100)

    def _load_zstack_from_folder(self):
        """Load z-stack from image folder (original behavior)."""
        folder = self.zstack_folder_var.get()
        if not folder:
            messagebox.showerror("Error", "Please select a z-stack folder")
            return

        folder = Path(folder)
        if not folder.exists():
            messagebox.showerror("Error", f"Folder not found: {folder}")
            return

        # Find images
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))

        image_files = sorted(set(image_files))

        if not image_files:
            messagebox.showerror("Error", "No images found in folder")
            return

        # Load positions from CSV if provided
        positions_file = self.positions_file_var.get()
        pos_dict = None

        if positions_file and Path(positions_file).exists():
            try:
                import pandas as pd
                df = pd.read_csv(positions_file)
                # Use first column as filename, second as position (ignore column names)
                filenames_col = df.iloc[:, 0].astype(str)
                positions_col = df.iloc[:, 1].astype(float)
                pos_dict = {}
                for fn, pos in zip(filenames_col, positions_col):
                    pos_dict[fn] = pos
                    # Also store without extension for flexible matching
                    pos_dict[Path(fn).stem] = pos
                self._append_stats_text(f"\nLoaded {len(df)} positions from: {Path(positions_file).name}")
            except Exception as e:
                self._append_stats_text(f"\n⚠️  Error loading CSV: {e}")

        # Load images with progress
        self.zstack_images = []
        self.zstack_filenames = []
        temp_positions = []

        n_files = len(image_files)

        for i, img_path in enumerate(image_files):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.zstack_images.append(img)
                self.zstack_filenames.append(img_path.name)

                # Try to get position from CSV (try full name, then stem)
                csv_pos = None
                if pos_dict:
                    if img_path.name in pos_dict:
                        csv_pos = pos_dict[img_path.name]
                    elif img_path.stem in pos_dict:
                        csv_pos = pos_dict[img_path.stem]

                if csv_pos is not None:
                    temp_positions.append(csv_pos)
                else:
                    # Fallback to interpolation
                    z_min = float(self.z_min_var.get())
                    z_max = float(self.z_max_var.get())
                    z = z_min + (z_max - z_min) * i / (n_files - 1) if n_files > 1 else 0
                    temp_positions.append(z)

            self.load_progress_var.set((i + 1) / n_files * 100)
            self.root.update_idletasks()

        # Apply focus offset to convert stage positions to defocus
        try:
            focus_offset = float(self.folder_focus_var.get())
        except ValueError:
            focus_offset = 0.0

        defocus_positions = [pos - focus_offset for pos in temp_positions]

        # Sort everything by position so slider goes in order
        if self.zstack_images:
            sorted_indices = np.argsort(defocus_positions)
            self.zstack_images = [self.zstack_images[i] for i in sorted_indices]
            self.zstack_filenames = [self.zstack_filenames[i] for i in sorted_indices]
            self.zstack_positions = [defocus_positions[i] for i in sorted_indices]
        else:
            self.zstack_positions = defocus_positions

        # Create stats
        if self.zstack_images:
            h, w = self.zstack_images[0].shape
            z_min = min(self.zstack_positions)
            z_max = max(self.zstack_positions)
            n_pos = len(self.zstack_positions)
            z_step = (z_max - z_min) / (n_pos - 1) if n_pos > 1 else 0

            self.zstack_stats = ZStackStats(
                num_images=len(self.zstack_images),
                image_width=w,
                image_height=h,
                z_min=z_min,
                z_max=z_max,
                z_step=z_step
            )

            # Update slider
            self.preview_slider.configure(to=len(self.zstack_images) - 1)
            self._update_preview(0)

            # Update stats display
            self._update_stats_text(f"Loaded Z-Stack\n" + "=" * 40)
            self._append_stats_text(f"\nImages: {self.zstack_stats.num_images}")
            self._append_stats_text(f"\nSize: {w} × {h} px")
            if focus_offset != 0:
                self._append_stats_text(f"\nFocus at position: {focus_offset:.2f} mm")
            self._append_stats_text(f"\nDefocus range: {self.zstack_stats.z_min:.1f} to {self.zstack_stats.z_max:.1f} mm")
            self._append_stats_text(f"\nZ step: {self.zstack_stats.z_step:.2f} mm")

        self.load_status_var.set(f"Loaded {len(self.zstack_images)} images")
        self.load_progress_var.set(100)

    def _on_preview_slider(self, value):
        idx = int(float(value))
        self._update_preview(idx)

    def _update_preview(self, idx: int):
        """Update the preview image."""
        if 0 <= idx < len(self.zstack_images):
            img = self.zstack_images[idx]
            z = self.zstack_positions[idx] if idx < len(self.zstack_positions) else 0

            self.preview_label_var.set(f"z = {z:.2f} mm\n(#{idx + 1} of {len(self.zstack_images)})")

            # Resize for display
            h, w = img.shape
            max_size = 450
            scale = min(max_size / w, max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h))

            # Convert to PhotoImage
            pil_img = Image.fromarray(img_resized)
            self._preview_photo = ImageTk.PhotoImage(pil_img)

            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(225, 225, image=self._preview_photo)

    def _auto_detect_focal(self):
        """Sanity check: verify sharpest image is at z=0."""
        if not self.zstack_images:
            messagebox.showerror("Error", "No images loaded")
            return

        sharpness = []
        for img in self.zstack_images:
            # Use uint8 directly - OpenCV supports uint8 -> CV_64F
            lap = cv2.Laplacian(img, cv2.CV_64F)
            sharpness.append(lap.var())

        best_idx = int(np.argmax(sharpness))
        best_z = self.zstack_positions[best_idx] if best_idx < len(self.zstack_positions) else 0

        if self.zstack_stats:
            self.zstack_stats.focal_plane_idx = best_idx
            self.zstack_stats.focal_plane_z = best_z

        self.preview_slider.set(best_idx)
        self._update_preview(best_idx)

        # Show result as sanity check
        if abs(best_z) < 0.3:  # Within 0.3mm of z=0
            self.focal_info_var.set(f"✓ Sharpest at z = {best_z:.2f} mm (looks good!)")
        else:
            self.focal_info_var.set(f"⚠ Sharpest at z = {best_z:.2f} mm (expected ~0)")
        self._append_stats_text(f"\n\nSanity check: sharpest image at z = {best_z:.2f} mm")

    def _auto_crop_to_sphere(self):
        """Auto-crop all images to sphere region with padding."""
        if not self.zstack_images:
            messagebox.showerror("Error", "No images loaded")
            return

        try:
            padding = int(self.crop_padding_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid padding value")
            return

        self.crop_status_var.set("Finding sharpest frame...")
        self.root.update_idletasks()

        # Find sharpest frame (best for detection)
        sharpness = [cv2.Laplacian(img, cv2.CV_64F).var() for img in self.zstack_images]
        best_idx = int(np.argmax(sharpness))

        self.crop_status_var.set("Detecting sphere...")
        self.root.update_idletasks()

        # Detect sphere in sharpest frame
        center, radius = detect_sphere(self.zstack_images[best_idx])
        if center is None:
            messagebox.showerror("Error", "Could not detect sphere in sharpest frame")
            self.crop_status_var.set("Detection failed")
            return

        cx, cy = center
        h, w = self.zstack_images[0].shape

        # Define square crop region (sphere diameter + padding on each side)
        crop_size = 2 * radius + 2 * padding

        # Calculate crop bounds centered on sphere
        x1 = max(0, cx - crop_size // 2)
        y1 = max(0, cy - crop_size // 2)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)

        # Adjust if crop goes out of bounds
        if x2 - x1 < crop_size:
            x1 = max(0, x2 - crop_size)
        if y2 - y1 < crop_size:
            y1 = max(0, y2 - crop_size)

        crop_w = x2 - x1
        crop_h = y2 - y1

        self.crop_status_var.set(f"Cropping {len(self.zstack_images)} images...")
        self.root.update_idletasks()

        # Apply same crop to all images
        cropped_images = []
        for i, img in enumerate(self.zstack_images):
            cropped = img[y1:y2, x1:x2].copy()
            cropped_images.append(cropped)
            self.load_progress_var.set((i + 1) / len(self.zstack_images) * 100)
            self.root.update_idletasks()

        # Replace images with cropped versions
        self.zstack_images = cropped_images

        # Update stats
        if self.zstack_stats:
            self.zstack_stats.image_width = crop_w
            self.zstack_stats.image_height = crop_h

        # Update display
        self._update_stats_text(
            f"Images: {len(self.zstack_images)}\n"
            f"Size: {crop_w} x {crop_h} (cropped)\n"
            f"Z range: {self.zstack_stats.z_min:.2f} to {self.zstack_stats.z_max:.2f} mm"
        )

        self._update_preview(int(self.preview_slider.get()))
        self.crop_status_var.set(f"Cropped to {crop_w}x{crop_h}")
        self.load_progress_var.set(0)

        messagebox.showinfo("Auto-Crop", f"Cropped {len(self.zstack_images)} images to {crop_w}x{crop_h} pixels\nCentered on sphere at ({cx}, {cy})")

    def _save_cropped_images(self):
        """Save cropped images to 'Cropped Z-Stack' subfolder next to script."""
        if not self.zstack_images:
            messagebox.showerror("Error", "No images loaded")
            return

        # Create "Cropped Z-Stack" subfolder next to this script (portable)
        script_dir = Path(__file__).parent
        output_path = script_dir / "Cropped Z-Stack"
        output_path.mkdir(parents=True, exist_ok=True)

        self.crop_status_var.set("Saving...")
        self.root.update_idletasks()

        # Save each image
        for i, (img, filename) in enumerate(zip(self.zstack_images, self.zstack_filenames)):
            if filename:
                out_name = filename
            else:
                out_name = f"crop_{i:04d}.png"

            out_path = output_path / out_name
            cv2.imwrite(str(out_path), img)

            self.load_progress_var.set((i + 1) / len(self.zstack_images) * 100)
            self.root.update_idletasks()

        self.load_progress_var.set(0)
        self.crop_status_var.set(f"Saved to Cropped Z-Stack/")
        messagebox.showinfo("Save Complete", f"Saved {len(self.zstack_images)} images to:\n{output_path}")

    def _detect_sphere_in_preview(self):
        """Detect sphere in current preview image."""
        idx = int(self.preview_slider.get())
        if 0 <= idx < len(self.zstack_images):
            center, radius = detect_sphere(self.zstack_images[idx])
            if center and radius:
                self.sphere_cx_var.set(str(center[0]))
                self.sphere_cy_var.set(str(center[1]))
                self.sphere_r_var.set(str(radius))

                if self.zstack_stats:
                    self.zstack_stats.detected_sphere_center = center
                    self.zstack_stats.detected_sphere_radius = radius

                self._append_stats_text(f"\n\nSphere detected:")
                self._append_stats_text(f"\n  Center: ({center[0]}, {center[1]})")
                self._append_stats_text(f"\n  Radius: {radius} px")

                # Draw on preview
                self._draw_sphere_overlay(center)
            else:
                messagebox.showinfo("Detection", "Could not detect sphere in current image")

    def _draw_sphere_overlay(self, center):
        """Draw sphere detection overlay on preview using actual mask outline."""
        idx = int(self.preview_slider.get())
        if 0 <= idx < len(self.zstack_images):
            img = self.zstack_images[idx].copy()
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Get actual mask and draw its contour (not a fitted circle)
            mask = get_sphere_mask(img)
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(img_color, contours, -1, (0, 255, 0), 2)

            # Draw centroid
            cv2.circle(img_color, center, 3, (0, 0, 255), -1)

            # Resize and display
            h, w = img_color.shape[:2]
            max_size = 450
            scale = min(max_size / w, max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img_color, (new_w, new_h))

            pil_img = Image.fromarray(img_resized)
            self._preview_photo = ImageTk.PhotoImage(pil_img)

            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(225, 225, image=self._preview_photo)

    def _update_stats_text(self, text):
        """Update stats text widget."""
        self.zstack_stats_text.configure(state='normal')
        self.zstack_stats_text.delete('1.0', 'end')
        self.zstack_stats_text.insert('1.0', text)
        self.zstack_stats_text.configure(state='disabled')

    def _append_stats_text(self, text):
        """Append to stats text widget."""
        self.zstack_stats_text.configure(state='normal')
        self.zstack_stats_text.insert('end', text)
        self.zstack_stats_text.configure(state='disabled')

    # =========================================================================
    # Event Handlers - Tab 2
    # =========================================================================
    def _on_auto_detect_toggle(self):
        """Toggle manual sphere entry state."""
        state = 'disabled' if self.auto_detect_var.get() else 'normal'
        self.sphere_cx_entry.configure(state=state)
        self.sphere_cy_entry.configure(state=state)
        self.sphere_r_entry.configure(state=state)

    def _measure_blur(self):
        """Measure blur for all z-stack images."""
        if not self.zstack_images:
            messagebox.showerror("Error", "No images loaded")
            return

        method = self.blur_method_var.get()

        # Get sphere parameters
        center = None
        radius = None
        if not self.auto_detect_var.get():
            try:
                cx = int(self.sphere_cx_var.get())
                cy = int(self.sphere_cy_var.get())
                radius = int(self.sphere_r_var.get())
                center = (cx, cy)
            except ValueError:
                pass

        # Run measurement in thread
        def measure_thread():
            self.sigma_values = []
            self.blur_measurements = []

            n = len(self.zstack_images)
            for i, img in enumerate(self.zstack_images):
                z = self.zstack_positions[i] if i < len(self.zstack_positions) else 0
                print(f"\n[{i+1}/{n}] Measuring image at z = {z:.2f} mm")
                measurement = measure_blur_auto(img, center, radius, method, verbose=True)
                print(f"  => σ = {measurement.sigma:.3f} px, confidence = {measurement.confidence:.3f}")
                self.sigma_values.append(measurement.sigma)
                self.blur_measurements.append(measurement)

                progress = (i + 1) / n * 100
                self.msg_queue.put(('measure_progress', progress))
                self.msg_queue.put(('measure_status', f"Measuring {i + 1}/{n}..."))

            self.msg_queue.put(('measure_done', None))

        self.measure_btn.configure(state='disabled')
        threading.Thread(target=measure_thread, daemon=True).start()

    def _measure_single(self):
        """Measure blur for current preview image only."""
        idx = int(self.preview_slider.get())
        if 0 <= idx < len(self.zstack_images):
            method = self.blur_method_var.get()

            center = None
            radius = None
            if not self.auto_detect_var.get():
                try:
                    center = (int(self.sphere_cx_var.get()), int(self.sphere_cy_var.get()))
                    radius = int(self.sphere_r_var.get())
                except ValueError:
                    pass

            z = self.zstack_positions[idx] if idx < len(self.zstack_positions) else 0
            print(f"\n[Single] Measuring image #{idx+1} at z = {z:.2f} mm")
            measurement = measure_blur_auto(self.zstack_images[idx], center, radius, method, verbose=True)
            print(f"  => σ = {measurement.sigma:.3f} px, confidence = {measurement.confidence:.3f}")

            messagebox.showinfo(
                "Single Measurement",
                f"Image #{idx + 1} (z = {z:.2f} mm)\n\n"
                f"σ = {measurement.sigma:.2f} px\n"
                f"Confidence: {measurement.confidence:.2f}\n"
                f"Method: {measurement.method}"
            )

    def _on_measure_complete(self):
        """Handle measurement completion."""
        self.measure_btn.configure(state='normal')
        self.measure_status_var.set(f"Done - {len(self.sigma_values)} measurements")

        # Update summary
        valid_sigmas = [s for s in self.sigma_values if not np.isnan(s)]
        if valid_sigmas:
            summary = f"Measurement Summary\n" + "=" * 30
            summary += f"\nTotal images: {len(self.sigma_values)}"
            summary += f"\nValid measurements: {len(valid_sigmas)}"
            summary += f"\nσ range: {min(valid_sigmas):.2f} - {max(valid_sigmas):.2f} px"

            # Find σ at z closest to 0
            if self.zstack_positions and len(self.zstack_positions) == len(self.sigma_values):
                idx_closest = np.argmin(np.abs(np.array(self.zstack_positions)))
                sigma_at_focus = self.sigma_values[idx_closest]
                z_at_focus = self.zstack_positions[idx_closest]
                if not np.isnan(sigma_at_focus):
                    summary += f"\nσ at z={z_at_focus:.2f}mm: {sigma_at_focus:.2f} px"

            self.measure_summary_text.configure(state='normal')
            self.measure_summary_text.delete('1.0', 'end')
            self.measure_summary_text.insert('1.0', summary)
            self.measure_summary_text.configure(state='disabled')

        # Update plot (use shared calib_ax)
        if HAS_MATPLOTLIB:
            self.calib_ax.clear()
            # Match lengths safely
            n = min(len(self.zstack_positions), len(self.sigma_values))
            z = self.zstack_positions[:n]
            sigma = self.sigma_values[:n]
            self.calib_ax.scatter(z, sigma, c='blue', alpha=0.7, s=30, label='Measured')
            self.calib_ax.set_xlabel('Defocus z (mm)')
            self.calib_ax.set_ylabel('Blur σ (pixels)')
            self.calib_ax.set_title('Blur vs Defocus')
            self.calib_ax.grid(True, alpha=0.3)
            self.calib_canvas.draw()

        # Update table
        for item in self.measure_tree.get_children():
            self.measure_tree.delete(item)

        for z, sigma, meas in zip(self.zstack_positions, self.sigma_values, self.blur_measurements):
            self.measure_tree.insert('', 'end', values=(
                f"{z:.2f}",
                f"{sigma:.2f}" if not np.isnan(sigma) else "N/A",
                f"{meas.confidence:.2f}"
            ))

    # =========================================================================
    # Event Handlers - Tab 3
    # =========================================================================
    def _on_calibration_mode_change(self):
        """Handle calibration mode change between optical and direct."""
        mode = self.calibration_mode_var.get()

        if mode == "optical":
            # Show ref_row (needed for optical mode with hybrid approach)
            self.ref_row.pack(fill='x', pady=1)
            print("Calibration Mode: Optical Formula")

        else:  # direct mode
            # Hide ref_row (not needed for direct mode)
            self.ref_row.pack_forget()
            print("Calibration Mode: Direct Calibration")

    def _run_calibration(self):
        """Run the calibration."""
        if not self.sigma_values or not self.zstack_positions:
            messagebox.showerror("Error", "No blur measurements available. Run measurement first.")
            return

        approach = self.approach_var.get()

        # Filter out failed measurements (σ = 0 means fitting failed completely)
        filtered_positions = []
        filtered_sigmas = []

        for z, sigma in zip(self.zstack_positions, self.sigma_values):
            if sigma > 0:  # Only exclude actual failures (σ = 0)
                filtered_positions.append(z)
                filtered_sigmas.append(sigma)

        n_filtered = len(self.sigma_values) - len(filtered_sigmas)
        if n_filtered > 0:
            self._append_stats_text(f"\n\nFiltered out {n_filtered} failed measurements (σ = 0)")

        if len(filtered_sigmas) < 5:
            messagebox.showerror("Error", f"Only {len(filtered_sigmas)} valid measurements after filtering. Need at least 5.")
            return

        # Store filtered data for plotting
        self.filtered_positions = filtered_positions
        self.filtered_sigmas = filtered_sigmas

        try:
            # Clear all previous calibrations first
            self.calibration_b = None
            self.calibration_hybrid = None

            if approach == 'B':
                optical = OpticalParams(
                    focal_length_mm=float(self.optical_vars['focal_length'].get()),
                    f_number=float(self.optical_vars['f_number'].get()),
                    focus_distance_mm=float(self.optical_vars['focus_distance'].get()),
                    pixel_size_mm=float(self.optical_vars['pixel_size'].get())
                )
                self.calibration_b = calibrate_approach_b(filtered_positions, filtered_sigmas, optical)
                self._display_results_b()

            else:  # hybrid
                optical = OpticalParams(
                    focal_length_mm=float(self.optical_vars['focal_length'].get()),
                    f_number=float(self.optical_vars['f_number'].get()),
                    focus_distance_mm=float(self.optical_vars['focus_distance'].get()),
                    pixel_size_mm=float(self.optical_vars['pixel_size'].get())
                )
                ref_d = float(self.ref_defocus_var.get())
                self.calibration_hybrid = calibrate_hybrid(filtered_positions, filtered_sigmas, optical, ref_d)
                self._display_results_hybrid()

            self._update_calibration_plot()
            self._update_fit_quality()
            self._update_export_preview()

        except Exception as e:
            messagebox.showerror("Calibration Error", str(e))

    def _display_results_b(self):
        """Display Known Optics results."""
        r = self.calibration_b
        o = r.optical_params
        text = f"""{'=' * 50}
KNOWN OPTICS - Full CoC Formula
{'=' * 50}

Your exact optical parameters:
  f = {o.focal_length_mm} mm
  N = f/{o.f_number}
  D = {o.focus_distance_mm} mm
  pixel = {o.pixel_size_mm} mm

RESULTS:
  ρ = {r.rho:.4f} (should be ≈1.0 if params are accurate)
  R² = {r.r_squared:.4f}
  Points used: {r.num_points}

FOR TRAINING GUI:
  Enter the optical params above
  Set ρ = {r.rho:.4f}
"""
        self._set_results_text(text)

    def _display_results_hybrid(self):
        """Display Estimated Optics results."""
        h = self.calibration_hybrid
        a = h.direct_result
        b = h.formula_result
        o = b.optical_params

        text = f"""{'=' * 50}
ESTIMATED OPTICS - Best of Both
{'=' * 50}

Fitted from data:
  ρ = {a.rho_px_per_mm:.4f} px/mm
  σ₀ = {a.sigma_0:.2f} px (blur at focus)
  R² = {a.r_squared:.4f}

Converted for Training GUI:
  Reference defocus: {h.conversion_reference_d} mm
  ρ = {b.rho:.4f} (compensates for estimation errors)

{'=' * 50}
FOR TRAINING GUI - COPY THESE VALUES:
{'=' * 50}
  focal_length_mm: {o.focal_length_mm}
  f_number: {o.f_number}
  focus_distance_mm: {o.focus_distance_mm}
  pixel_size_mm: {o.pixel_size_mm}
  rho: {b.rho:.4f}

The ρ value compensates for any errors in your estimates!
"""
        self._set_results_text(text)

    def _set_results_text(self, text):
        """Set results text widget content."""
        self.results_text.configure(state='normal')
        self.results_text.delete('1.0', 'end')
        self.results_text.insert('1.0', text)
        self.results_text.configure(state='disabled')

    def _update_fit_quality(self):
        """Update fit quality display."""
        if self.calibration_hybrid:
            a = self.calibration_hybrid.direct_result
            is_valid, warnings = validate_calibration(a)

            quality = f"Fit Quality Assessment\n" + "=" * 30
            quality += f"\n\nR² = {a.r_squared:.4f}"
            quality += " ✓ Good" if a.r_squared > 0.95 else " ⚠️ Fair" if a.r_squared > 0.9 else " ❌ Poor"
            quality += f"\nσ₀ = {a.sigma_0:.2f} px"
            quality += " ✓" if a.sigma_0 < 3 else " ⚠️ High"

            if warnings:
                quality += "\n\nWarnings:"
                for w in warnings:
                    quality += f"\n  • {w}"
            else:
                quality += "\n\n✓ Calibration looks good!"

            self.fit_quality_text.configure(state='normal')
            self.fit_quality_text.delete('1.0', 'end')
            self.fit_quality_text.insert('1.0', quality)
            self.fit_quality_text.configure(state='disabled')

    def _validate_fit(self):
        """Validate the current fit."""
        if not self.calibration_hybrid and not self.calibration_b:
            messagebox.showerror("Error", "Run calibration first")
            return

        if self.calibration_hybrid:
            result = self.calibration_hybrid.direct_result
        else:
            # For Known Optics (B), create a minimal result object for validation
            result = self.calibration_b
        is_valid, warnings = validate_calibration(result)

        if is_valid:
            messagebox.showinfo("Validation", "✓ Calibration passed all checks!")
        else:
            messagebox.showwarning("Validation", "Warnings:\n" + "\n".join(warnings))

    def _update_calibration_plot(self):
        """Update the calibration fit plot."""
        if not HAS_MATPLOTLIB:
            return

        self.calib_ax.clear()

        # Use filtered data if available, otherwise all data
        if hasattr(self, 'filtered_positions') and self.filtered_positions:
            z_valid = np.array(self.filtered_positions)
            sigma_valid = np.array(self.filtered_sigmas)

            # Also show filtered-out points in gray (σ = 0 means failed)
            z_all = np.array(self.zstack_positions)
            sigma_all = np.array(self.sigma_values)
            mask = sigma_all <= 0
            if np.any(mask):
                self.calib_ax.scatter(z_all[mask], sigma_all[mask], c='lightgray', alpha=0.5, s=20, label='Failed')
        else:
            z = np.array(self.zstack_positions)
            sigma = np.array(self.sigma_values)
            valid = ~np.isnan(sigma)
            z_valid = z[valid]
            sigma_valid = sigma[valid]

        # Plot data points
        self.calib_ax.scatter(z_valid, sigma_valid, c='blue', label='Measured', alpha=0.7, s=40)

        # Plot fit based on which calibration is active
        if self.calibration_hybrid:
            a = self.calibration_hybrid.direct_result
            z_fit = np.linspace(z_valid.min(), z_valid.max(), 100)
            sigma_fit = a.rho_px_per_mm * np.abs(z_fit) + a.sigma_0
            self.calib_ax.plot(z_fit, sigma_fit, 'r-', label=f'Estimated: ρ = {a.rho_px_per_mm:.3f} px/mm', linewidth=2)
        elif self.calibration_b:
            b = self.calibration_b
            z_fit = np.linspace(z_valid.min(), z_valid.max(), 100)
            # For Known Optics, compute sigma from CoC formula
            sigma_fit = np.array([b.rho * b.optical_params.calculate_coc(z) for z in z_fit])
            self.calib_ax.plot(z_fit, sigma_fit, 'r-', label=f'Known: ρ = {b.rho:.3f}', linewidth=2)

        self.calib_ax.set_xlabel('Defocus z (mm)')
        self.calib_ax.set_ylabel('Blur σ (pixels)')
        self.calib_ax.set_title('Calibration Curve: σ vs |z|')
        self.calib_ax.legend()
        self.calib_ax.grid(True, alpha=0.3)

        self.calib_canvas.draw()

    # =========================================================================
    # Event Handlers - Tab 3 (Multi-Camera)
    # =========================================================================
    def _get_current_calibration(self):
        """Get the current active calibration and its rho in px/mm."""
        if self.calibration_hybrid:
            rho = self.calibration_hybrid.direct_result.rho_px_per_mm
            return ('hybrid', self.calibration_hybrid, rho)
        elif self.calibration_b:
            # For approach B, estimate rho_px_per_mm at reference defocus
            ref_d = float(self.ref_defocus_var.get())
            coc = self.calibration_b.optical_params.calculate_coc(ref_d)
            rho_px_per_mm = (self.calibration_b.rho * coc) / ref_d if ref_d != 0 else 0
            return ('B', self.calibration_b, rho_px_per_mm)
        return (None, None, None)

    def _add_camera_calibration(self):
        """Add current calibration to camera list."""
        approach, calib, rho = self._get_current_calibration()
        if not calib:
            messagebox.showerror("Error", "Run calibration first (Tab 2)")
            return

        camera = self.camera_var.get()

        try:
            offset = float(self.focal_offset_var.get())
        except ValueError:
            offset = 0.0

        # First camera becomes reference
        if not self.camera_calibrations:
            offset = 0.0
            self.focal_offset_var.set("0.0")

        # Store calibration with its approach type and rho_px_per_mm
        self.camera_calibrations[camera] = {
            'approach': approach,
            'calibration': calib,
            'rho_px_per_mm': rho
        }
        self.focal_plane_offsets[camera] = offset

        self._refresh_camera_tree()

    def _remove_camera_calibration(self):
        """Remove selected camera calibration."""
        selection = self.camera_tree.selection()
        if selection:
            item = self.camera_tree.item(selection[0])
            camera = item['values'][0]
            if camera in self.camera_calibrations:
                del self.camera_calibrations[camera]
                del self.focal_plane_offsets[camera]
            self.camera_tree.delete(selection[0])

    def _refresh_camera_tree(self):
        """Refresh camera tree from stored calibrations."""
        for item in self.camera_tree.get_children():
            self.camera_tree.delete(item)

        for camera, data in self.camera_calibrations.items():
            rho = data['rho_px_per_mm']
            offset = self.focal_plane_offsets.get(camera, 0.0)
            aperture = self.aperture_var.get()

            self.camera_tree.insert('', 'end', values=(
                camera,
                f"{rho:.3f}",
                f"{offset:.1f}",
                aperture
            ))

    def _test_sign_resolution(self):
        """Test sign resolution with entered values."""
        if len(self.camera_calibrations) < 2:
            messagebox.showerror("Error", "Need at least 2 camera calibrations")
            return

        try:
            sigma1 = float(self.test_sigma1_var.get())
            sigma2 = float(self.test_sigma2_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid sigma values")
            return

        cameras = list(self.camera_calibrations.keys())
        cam1, cam2 = cameras[0], cameras[1]

        rho1 = self.camera_calibrations[cam1]['rho_px_per_mm']
        rho2 = self.camera_calibrations[cam2]['rho_px_per_mm']
        offset = self.focal_plane_offsets[cam2] - self.focal_plane_offsets[cam1]

        # Calculate depths
        d1 = sigma1 / rho1
        d2 = sigma2 / rho2

        # Average magnitude
        depth_mag = (d1 + d2) / 2

        # Determine sign
        if offset > 0:
            sign = +1 if sigma2 < sigma1 else -1
        else:
            sign = -1 if sigma2 < sigma1 else +1

        signed_depth = sign * depth_mag
        sign_str = "+" if sign > 0 else ""

        self.sign_result_var.set(f"Depth = {sign_str}{signed_depth:.1f} mm")

        # Explanation
        if sigma2 < sigma1:
            sharper = cam2
        else:
            sharper = cam1

        explanation = f"Camera {sharper} sees sharper image (lower σ).\n"
        if offset > 0:
            explanation += f"Since {cam2}'s focal plane is {abs(offset):.1f}mm behind {cam1}'s,\n"
        else:
            explanation += f"Since {cam2}'s focal plane is {abs(offset):.1f}mm in front of {cam1}'s,\n"

        explanation += f"the droplet is {'behind' if sign > 0 else 'in front of'} {cam1}'s focal plane."

        self.sign_explanation_var.set(explanation)

    # =========================================================================
    # Event Handlers - Tab 6 (Export)
    # =========================================================================
    def _browse_export_folder(self):
        folder = filedialog.askdirectory(title="Select Export Folder")
        if folder:
            self.export_folder_var.set(folder)

    def _update_export_preview(self):
        """Update export preview."""
        if not self.calibration_hybrid:
            return

        # Get defocus range from z-stack positions
        defocus_range = None
        if self.zstack_positions and len(self.zstack_positions) > 0:
            defocus_range = (min(self.zstack_positions), max(self.zstack_positions))

        # Get reference resolution from loaded images
        reference_resolution = None
        if self.zstack_stats:
            reference_resolution = max(self.zstack_stats.image_width, self.zstack_stats.image_height)

        yaml_dict = export_calibration_yaml(
            self.calibration_hybrid,
            camera=self.camera_var.get(),
            aperture_setting=self.aperture_var.get(),
            focal_plane_offset_mm=float(self.focal_offset_var.get()) if self.focal_offset_var.get() else 0.0,
            defocus_range_mm=defocus_range,
            reference_resolution=reference_resolution,
            calibration_mode=self.calibration_mode_var.get()
        )

        yaml_str = yaml.dump(yaml_dict, default_flow_style=False, sort_keys=False)

        self.export_preview.configure(state='normal')
        self.export_preview.delete('1.0', 'end')
        self.export_preview.insert('1.0', yaml_str)
        self.export_preview.configure(state='disabled')

    def _export_calibration(self):
        """Export calibration results."""
        if not self.calibration_hybrid:
            messagebox.showerror("Error", "Run calibration first")
            return

        output_dir = Path(self.export_folder_var.get())
        output_dir.mkdir(parents=True, exist_ok=True)

        exported = []

        # Export YAML
        if self.export_yaml_var.get():
            # Get defocus range from z-stack positions
            defocus_range = None
            if self.zstack_positions and len(self.zstack_positions) > 0:
                defocus_range = (min(self.zstack_positions), max(self.zstack_positions))

            # Get reference resolution from loaded images
            reference_resolution = None
            if self.zstack_stats:
                reference_resolution = max(self.zstack_stats.image_width, self.zstack_stats.image_height)

            yaml_dict = export_calibration_yaml(
                self.calibration_hybrid,
                camera=self.camera_var.get(),
                aperture_setting=self.aperture_var.get(),
                focal_plane_offset_mm=float(self.focal_offset_var.get()) if self.focal_offset_var.get() else 0.0,
                defocus_range_mm=defocus_range,
                reference_resolution=reference_resolution,
                calibration_mode=self.calibration_mode_var.get()
            )
            yaml_path = output_dir / "calibration_results.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)
            exported.append("calibration_results.yaml")

        # Export CSV
        if self.export_csv_var.get():
            import pandas as pd
            df = pd.DataFrame({
                'z_mm': self.zstack_positions[:len(self.sigma_values)],
                'sigma_px': self.sigma_values
            })
            csv_path = output_dir / "measurements.csv"
            df.to_csv(csv_path, index=False)
            exported.append("measurements.csv")

        # Export plots
        if self.export_plots_var.get() and HAS_MATPLOTLIB:
            plot_path = output_dir / "calibration_curve.png"
            self.calib_fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            exported.append("calibration_curve.png")

        self.export_status_var.set(f"✓ Exported to {output_dir.name}/")
        messagebox.showinfo("Export Complete", f"Calibration exported to:\n{output_dir}\n\nFiles:\n" + "\n".join(exported))

    def _copy_rho(self):
        """Copy ρ value to clipboard."""
        if self.calibration_hybrid:
            rho = self.calibration_hybrid.formula_result.rho
            self.root.clipboard_clear()
            self.root.clipboard_append(f"{rho:.4f}")
            messagebox.showinfo("Copied", f"ρ = {rho:.4f} copied to clipboard\n\nPaste this in Training GUI's 'ρ (blur constant)' field")
        else:
            messagebox.showerror("Error", "Run calibration first")

    def _lookup_pixel_size(self):
        """Open browser to search for camera pixel size datasheet."""
        # Try to get camera name from loaded .cine files
        camera_name = None
        if self.cine_folder_loader and self.cine_folder_loader.cine_files:
            # Try to extract camera info from first cine file
            try:
                from cine_loader import CineLoader
                first_cine = self.cine_folder_loader.cine_files[0]
                loader = CineLoader(str(first_cine))
                if loader.cine_obj is not None:
                    # pyphantom may have camera info in cine_obj attributes
                    # Try common attribute names
                    for attr in ['camera', 'cameraModel', 'camera_model', 'CameraModel']:
                        if hasattr(loader.cine_obj, attr):
                            camera_name = getattr(loader.cine_obj, attr)
                            break
            except Exception:
                pass

        # Build search query
        if camera_name:
            query = f"{camera_name} datasheet pixel size"
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(search_url)
        else:
            # No camera detected - open generic search with placeholder
            query = "[enter camera name] datasheet pixel size"
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(search_url)

    # =========================================================================
    # Message Processing
    # =========================================================================
    def _process_messages(self):
        """Process messages from background threads."""
        try:
            while True:
                msg_type, data = self.msg_queue.get_nowait()

                if msg_type == 'measure_progress':
                    self.measure_progress_var.set(data)
                elif msg_type == 'measure_status':
                    self.measure_status_var.set(data)
                elif msg_type == 'measure_done':
                    self._on_measure_complete()

        except queue.Empty:
            pass

        self.root.after(100, self._process_messages)

    def run(self):
        """Run the application."""
        self.root.mainloop()


# =============================================================================
# Main
# =============================================================================
def main():
    """Run the calibration GUI."""
    root = tk.Tk()
    app = CalibrationGUI(root)
    app.run()


if __name__ == "__main__":
    main()
