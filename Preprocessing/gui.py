"""GUI interface for the droplet cropping pipeline."""

import builtins
import queue
import threading
import time
import subprocess
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk
except ImportError:
    print("GUI requires: pip install pillow")
    raise

import config
from cine_io import cine
from profiling import format_time

# Queue for worker -> GUI communication (logs, progress, done signals)
gui_queue: queue.Queue = queue.Queue()


def emit_log(message: str) -> None:
    """Send log message to GUI."""
    gui_queue.put(("log", message))


def emit_progress(current: int, total: int) -> None:
    """Send progress update to GUI."""
    gui_queue.put(("progress", (current, total)))


def emit_done() -> None:
    """Signal processing complete."""
    gui_queue.put(("done", None))


class PipelineGUI:
    """Main GUI window with landing + main screen."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Droplet Cropping Pipeline")
        self.root.geometry("800x700")
        self.root.minsize(700, 600)

        # Single cell layout for swapping frames
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # State
        self.processing: bool = False
        self.worker_thread: Optional[threading.Thread] = None
        self.thumbnails: List[ttk.Label] = []
        self.thumbnail_paths: List[str] = []
        self.thumbnail_images: List[Optional[ImageTk.PhotoImage]] = []  # Keep references
        self.selected_root: Optional[Path] = None
        self.start_time: Optional[float] = None
        self.progress_current: int = 0
        self.progress_total: int = 0

        # For output folder polling
        self.known_images: set = set()
        self.polling_active: bool = False

        # For elapsed timer
        self.elapsed_timer_active: bool = False

        # Track if selected folder has subfolders (for global mode availability)
        self.has_subfolders: bool = True

        # Top-level frames
        self.landing_frame = ttk.Frame(self.root)
        self.landing_frame.grid(row=0, column=0, sticky="nsew")

        # Main frame with scrollable canvas
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Create canvas and scrollbar for scrollable content
        self.main_canvas = tk.Canvas(self.main_frame, highlightthickness=0)
        self.main_scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set)

        self.main_scrollbar.grid(row=0, column=1, sticky="ns")
        self.main_canvas.grid(row=0, column=0, sticky="nsew")

        # Inner frame that holds all content
        self.main_inner = ttk.Frame(self.main_canvas)
        self.main_canvas_window = self.main_canvas.create_window((0, 0), window=self.main_inner, anchor="nw")

        # Configure canvas scrolling
        self.main_inner.bind("<Configure>", self._on_main_inner_configure)
        self.main_canvas.bind("<Configure>", self._on_main_canvas_configure)

        # Bind mousewheel scrolling
        self.main_canvas.bind("<Enter>", lambda e: self._bind_mousewheel())
        self.main_canvas.bind("<Leave>", lambda e: self._unbind_mousewheel())

        # Build UI inside scrollable inner frame
        self._build_landing()
        self._build_header(self.main_inner)
        self._build_config(self.main_inner)
        self._build_preview(self.main_inner)
        self._build_controls(self.main_inner)
        self._build_log(self.main_inner)

        # Start queue polling
        self.root.after(100, self._poll_queue)

        # Show landing first
        self._show_landing()

    def mainloop(self) -> None:
        """Start the main event loop."""
        self.root.mainloop()

    # --- Scrolling helpers ---
    def _on_main_inner_configure(self, event) -> None:
        """Update scroll region when inner frame size changes."""
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _on_main_canvas_configure(self, event) -> None:
        """Update inner frame width when canvas is resized."""
        self.main_canvas.itemconfig(self.main_canvas_window, width=event.width)

    def _bind_mousewheel(self) -> None:
        """Bind mousewheel to scroll."""
        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self) -> None:
        """Unbind mousewheel."""
        self.main_canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event) -> None:
        """Handle mousewheel scrolling."""
        self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # --- Landing screen ---
    def _build_landing(self) -> None:
        """Build landing page with CINE root and Output folder selection."""
        self.landing_frame.grid_columnconfigure(0, weight=1)
        self.landing_frame.grid_rowconfigure(0, weight=1)

        inner = ttk.Frame(self.landing_frame, padding=20)
        inner.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        inner.grid_columnconfigure(0, weight=1)

        # Title
        title = ttk.Label(
            inner,
            text="DROPLET CROPPING PIPELINE",
            font=("TkDefaultFont", 16, "bold"),
        )
        title.grid(row=0, column=0, pady=(30, 20), sticky="n")

        # ----- CINE Root Section -----
        cine_section = ttk.LabelFrame(inner, text="CINE Root", padding=10)
        cine_section.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        cine_section.grid_columnconfigure(0, weight=1)

        self.landing_cine_path = ttk.Label(
            cine_section,
            text=str(config.CINE_ROOT),
            wraplength=450,
            anchor="w",
        )
        self.landing_cine_path.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        ttk.Button(
            cine_section,
            text="Browse",
            width=10,
            command=self._select_cine_folder,
        ).grid(row=0, column=1, padx=10, pady=5, sticky="e")

        self.landing_cine_info = ttk.Label(
            cine_section,
            text="",
            foreground="gray",
        )
        self.landing_cine_info.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="w")

        # ----- Output Folder Section -----
        output_section = ttk.LabelFrame(inner, text="Output Folder", padding=10)
        output_section.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        output_section.grid_columnconfigure(0, weight=1)

        self.landing_output_path = ttk.Label(
            output_section,
            text=str(config.OUTPUT_ROOT),
            wraplength=350,
            anchor="w",
        )
        self.landing_output_path.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        ttk.Button(
            output_section,
            text="Use Default (./OUTPUT)",
            command=self._use_default_output,
        ).grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(
            output_section,
            text="Browse",
            width=10,
            command=self._select_output_folder,
        ).grid(row=0, column=2, padx=10, pady=5, sticky="e")

        # ----- Buttons Row -----
        btn_frame = ttk.Frame(inner)
        btn_frame.grid(row=3, column=0, pady=(30, 30))

        self.continue_btn = ttk.Button(
            btn_frame,
            text="Continue →",
            command=self._continue_to_main,
        )
        self.continue_btn.grid(row=0, column=0, padx=10)

        about_btn = ttk.Button(
            btn_frame,
            text="About",
            command=self._show_about,
        )
        about_btn.grid(row=0, column=1, padx=10)

    def _show_about(self) -> None:
        """Show About dialog with license information."""
        about_text = (
            "Droplet Preprocessing Pipeline\n\n"
            "Copyright (C) 2025 Justice Ward\n\n"
            "This program is free software: you can redistribute it and/or modify "
            "it under the terms of the GNU General Public License as published by "
            "the Free Software Foundation, either version 3 of the License, or "
            "(at your option) any later version.\n\n"
            "This program is distributed in the hope that it will be useful, "
            "but WITHOUT ANY WARRANTY; without even the implied warranty of "
            "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the "
            "GNU General Public License for more details.\n\n"
            "You should have received a copy of the GNU General Public License "
            "along with this program. If not, see <https://www.gnu.org/licenses/>."
        )
        messagebox.showinfo("About", about_text)

    def _show_landing(self) -> None:
        """Show landing screen."""
        self.main_frame.grid_remove()
        self.landing_frame.grid()
        self.landing_frame.tkraise()

    def _show_main(self) -> None:
        """Show main config/processing screen."""
        self.landing_frame.grid_remove()
        self.main_frame.grid()
        self.main_frame.tkraise()

    def _select_cine_folder(self) -> None:
        """Select CINE root folder from landing page."""
        folder = filedialog.askdirectory(
            title="Select folder containing .cine files"
        )
        if not folder:
            return

        root = Path(folder)
        if not root.exists():
            messagebox.showerror("Invalid folder", "Selected folder does not exist.")
            return

        config.CINE_ROOT = root
        self.selected_root = root
        self.landing_cine_path.configure(text=str(root))
        self._scan_cine_folder()

    def _scan_cine_folder(self) -> None:
        """Scan CINE folder and update info label."""
        root = config.CINE_ROOT
        try:
            from cine_io import get_cine_folders, iter_subfolders, group_cines_by_droplet

            cine_folders = get_cine_folders(root)
            n_folders = len(cine_folders)
            droplets = 0

            for folder in cine_folders:
                groups = group_cines_by_droplet(folder)
                droplets += len(groups)

            self.has_subfolders = len(iter_subfolders(root)) > 0 and n_folders > 1

            if n_folders > 0:
                text = f"{n_folders} folder{'s' if n_folders != 1 else ''}, ~{droplets} droplets"
            else:
                text = "No .cine files found"

        except Exception as e:
            text = f"Scan error: {e}"
            self.has_subfolders = False

        self.landing_cine_info.configure(text=text)

    def _select_output_folder(self) -> None:
        """Select custom output folder."""
        folder = filedialog.askdirectory(
            title="Select output folder"
        )
        if not folder:
            return

        root = Path(folder)
        config.OUTPUT_ROOT = root
        root.mkdir(parents=True, exist_ok=True)
        self.landing_output_path.configure(text=str(root))

    def _use_default_output(self) -> None:
        """Set output folder to default ./OUTPUT."""
        default = config.PROJECT_ROOT / "OUTPUT"
        config.OUTPUT_ROOT = default
        default.mkdir(parents=True, exist_ok=True)
        self.landing_output_path.configure(text=str(default))

    def _continue_to_main(self) -> None:
        """Validate selections and proceed to main screen."""
        # Check CINE root has files
        from cine_io import get_cine_folders
        cine_folders = get_cine_folders(config.CINE_ROOT)

        if not cine_folders:
            messagebox.showwarning(
                "No CINE files",
                "Selected CINE root contains no .cine files.\n\n"
                "Please select a folder containing .cine files or subfolders with .cine files."
            )
            return

        # Ensure output folder exists
        config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        # Update main screen header
        self.source_label.configure(text=str(config.CINE_ROOT))
        self._rescan_counts_for_root(config.CINE_ROOT)

        # Go to main screen
        self._show_main()

    def _change_cine_folder(self) -> None:
        """Change CINE root folder from main screen header."""
        folder = filedialog.askdirectory(
            title="Select folder containing .cine files"
        )
        if not folder:
            return

        root = Path(folder)
        if not root.exists():
            messagebox.showerror("Invalid folder", "Selected folder does not exist.")
            return

        config.CINE_ROOT = root
        self.selected_root = root

        # Update displays
        self.landing_cine_path.configure(text=str(root))
        self.source_label.configure(text=str(root))

        # Rescan
        self._rescan_counts_for_root(root)

    def _rescan_counts_for_root(self, root: Path) -> None:
        """Scan cine root and update counts on both landing + header."""
        try:
            from cine_io import get_cine_folders, iter_subfolders, group_cines_by_droplet

            cine_folders = get_cine_folders(root)
            n_folders = len(cine_folders)
            droplets = 0

            for folder in cine_folders:
                groups = group_cines_by_droplet(folder)
                droplets += len(groups)

            # Check if we have actual subfolders (for global mode availability)
            # Global mode only makes sense with multiple folders
            self.has_subfolders = len(iter_subfolders(root)) > 0 and n_folders > 1

            text = f"{n_folders} folder{'s' if n_folders != 1 else ''}, ~{droplets} droplets"

        except Exception as e:
            text = f"Scan error: {e}"
            self.has_subfolders = False

        self.landing_cine_info.configure(text=text)
        self.count_label.configure(text=text)

        # Update config state (may need to disable global mode)
        self._update_config_state()

    # --- Header ---
    def _build_header(self, parent: ttk.Frame) -> None:
        """Build header section."""
        frame = ttk.LabelFrame(parent, text="Pipeline", padding=10)
        frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        title = ttk.Label(
            frame,
            text="DROPLET CROPPING PIPELINE",
            font=("TkDefaultFont", 14, "bold"),
        )
        title.grid(row=0, column=0, columnspan=3, pady=(5, 10))

        # Source info
        ttk.Label(frame, text="Source:").grid(row=1, column=0, padx=10, sticky="w")
        default_root = config.CINE_ROOT
        self.source_label = ttk.Label(frame, text=str(default_root))
        self.source_label.grid(row=1, column=1, padx=10, sticky="w")

        change_btn = ttk.Button(
            frame,
            text="Change...",
            command=self._change_cine_folder,
        )
        change_btn.grid(row=1, column=2, padx=10, sticky="e")

        # Count info
        ttk.Label(frame, text="Found:").grid(row=2, column=0, padx=10, sticky="w")
        self.count_label = ttk.Label(frame, text="Scanning...")
        self.count_label.grid(row=2, column=1, padx=10, sticky="w", pady=(0, 5))

        # Check SDK + initial count in background
        threading.Thread(target=self._check_sdk, daemon=True).start()

    def _check_sdk(self) -> None:
        """Check Photron SDK availability and update counts."""
        try:
            _ = cine  # imported at top
            sdk_ok = True
        except Exception:
            sdk_ok = False

        # Basic cine count from whichever root is current
        root = self.selected_root or config.CINE_ROOT
        try:
            from cine_io import get_cine_folders, group_cines_by_droplet

            cine_folders = get_cine_folders(root)
            n_folders = len(cine_folders)
            droplets = 0
            for sub in cine_folders:
                droplets += len(group_cines_by_droplet(sub))

            text = f"{n_folders} folder{'s' if n_folders != 1 else ''}, ~{droplets} droplets"
        except Exception as e:
            text = f"Scan error: {e}"

        if not sdk_ok:
            text += "  (SDK NOT FOUND!)"

        def update() -> None:
            self.count_label.configure(text=text)

        self.root.after(0, update)

    # --- Config panel ---
    def _build_config(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        # ----- Execution mode -----
        exec_frame = ttk.LabelFrame(frame, text="Execution mode", padding=10)
        exec_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        exec_frame.grid_columnconfigure(0, weight=1)

        self.mode_var = tk.StringVar(value="full")

        self.quick_radio = ttk.Radiobutton(
            exec_frame,
            text="Quick detection test (1st droplet per folder)",
            variable=self.mode_var,
            value="quick",
            command=self._update_config_state,
        )
        self.quick_radio.grid(row=0, column=0, padx=5, pady=2, sticky="w")

        self.full_radio = ttk.Radiobutton(
            exec_frame,
            text="Full pipeline",
            variable=self.mode_var,
            value="full",
            command=self._update_config_state,
        )
        self.full_radio.grid(row=1, column=0, padx=5, pady=2, sticky="w")

        # Safe mode
        self.safe_var = tk.BooleanVar(value=False)
        self.safe_check = ttk.Checkbutton(
            exec_frame,
            text="Safe mode (single process, more stable)",
            variable=self.safe_var,
        )
        self.safe_check.grid(row=2, column=0, padx=5, pady=(5, 2), sticky="w")

        # ----- Sampling -----
        sample_frame = ttk.LabelFrame(frame, text="Sampling (cine step)", padding=10)
        sample_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        sample_frame.grid_columnconfigure(0, weight=1)

        self.sample_var = tk.StringVar(value="10")

        s_row = 0
        self.sample_radios: List[ttk.Radiobutton] = []
        step_labels = [
            ("1", "Every cine"),
            ("2", "Every 2nd cine"),
            ("5", "Every 5th cine"),
            ("10", "Every 10th cine"),
        ]
        for step, label in step_labels:
            rb = ttk.Radiobutton(
                sample_frame,
                text=label,
                variable=self.sample_var,
                value=step,
            )
            rb.grid(row=s_row, column=0, padx=5, pady=1, sticky="w")
            self.sample_radios.append(rb)
            s_row += 1

        # Custom step
        custom_frame = ttk.Frame(sample_frame)
        custom_frame.grid(row=s_row, column=0, padx=5, pady=(4, 5), sticky="w")
        self.custom_step = ttk.Entry(custom_frame, width=8)
        self.custom_step.grid(row=0, column=0, padx=(0, 5))
        self.custom_radio = ttk.Radiobutton(
            custom_frame,
            text="Every Nth cine",
            variable=self.sample_var,
            value="custom",
        )
        self.custom_radio.grid(row=0, column=1, padx=(0, 5), sticky="w")

        self.sampling_controls: List[tk.Widget] = (
            self.sample_radios + [self.custom_step, self.custom_radio]
        )

        # ----- Calibration & Outputs (second row) -----
        calib_frame = ttk.LabelFrame(frame, text="Calibration (crop)", padding=10)
        calib_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        calib_frame.grid_columnconfigure(0, weight=1)

        self.calib_var = tk.StringVar(value="folder")

        self.calib_folder_radio = ttk.Radiobutton(
            calib_frame,
            text="Per-folder crop (each folder calibrated independently)",
            variable=self.calib_var,
            value="folder",
        )
        self.calib_folder_radio.grid(row=0, column=0, padx=5, pady=2, sticky="w")

        self.calib_global_radio = ttk.Radiobutton(
            calib_frame,
            text="Global crop (calibration from all folders combined)",
            variable=self.calib_var,
            value="global",
        )
        self.calib_global_radio.grid(row=1, column=0, padx=5, pady=2, sticky="w")

        self.calib_controls = [self.calib_folder_radio, self.calib_global_radio]

        # Outputs + profiling
        output_frame = ttk.LabelFrame(frame, text="Outputs", padding=10)
        output_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        output_frame.grid_columnconfigure(0, weight=1)

        self.output_var = tk.StringVar(value="crops")

        self.output_crops_radio = ttk.Radiobutton(
            output_frame,
            text="Crops only (fastest)",
            variable=self.output_var,
            value="crops",
        )
        self.output_crops_radio.grid(row=0, column=0, padx=5, pady=2, sticky="w")

        self.output_all_radio = ttk.Radiobutton(
            output_frame,
            text="All plots",
            variable=self.output_var,
            value="all",
        )
        self.output_all_radio.grid(row=1, column=0, padx=5, pady=2, sticky="w")

        self.output_controls = [self.output_crops_radio, self.output_all_radio]

        self.profile_var = tk.BooleanVar(value=False)
        self.profile_check = ttk.Checkbutton(
            output_frame,
            text="Enable profiling",
            variable=self.profile_var,
        )
        self.profile_check.grid(row=2, column=0, padx=5, pady=(5, 2), sticky="w")

        self.focus_class_var = tk.BooleanVar(value=True)
        self.focus_class_check = ttk.Checkbutton(
            output_frame,
            text="Focus classification (per-folder)",
            variable=self.focus_class_var,
        )
        self.focus_class_check.grid(row=3, column=0, padx=5, pady=2, sticky="w")

        # Keep profiling and focus classification in same disable group as outputs
        self.profile_control = self.profile_check
        self.focus_class_control = self.focus_class_check

        # Initialise enable/disable state
        self._update_config_state()

    def _update_config_state(self) -> None:
        """Enable/disable controls depending on quick/full mode and subfolder availability."""
        quick = self.mode_var.get() == "quick"
        state = "disabled" if quick else "normal"

        # Sampling
        for w in self.sampling_controls:
            self._set_widget_state(w, state)

        # Calibration - also check for subfolders
        # Per-folder always available when not in quick mode
        self._set_widget_state(self.calib_folder_radio, state)

        # Global only available if we have subfolders AND not in quick mode
        if quick or not self.has_subfolders:
            global_state = "disabled"
            # If global is currently selected but unavailable, switch to folder
            if self.calib_var.get() == "global":
                self.calib_var.set("folder")
        else:
            global_state = "normal"
        self._set_widget_state(self.calib_global_radio, global_state)

        # Outputs
        for w in self.output_controls:
            self._set_widget_state(w, state)

        # Profiling and focus classification
        self._set_widget_state(self.profile_control, state)
        self._set_widget_state(self.focus_class_control, state)

    def _set_widget_state(self, widget: tk.Widget, state: str) -> None:
        """Set widget state to 'normal' or 'disabled'."""
        try:
            widget.configure(state=state)
        except Exception:
            pass

    # --- Preview thumbnails ---
    def _build_preview(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Recent outputs (click to open)", padding=10)
        frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(0, weight=1)

        thumb_row = ttk.Frame(frame)
        thumb_row.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # 5 thumbnails (latest on the right)
        for i in range(5):
            lbl = ttk.Label(
                thumb_row,
                text="",
                width=15,
                relief="sunken",
                anchor="center",
            )
            lbl.grid(row=0, column=i, padx=4, pady=4)
            # Set a minimum size using a blank image
            blank = Image.new("RGB", (110, 110), color=(200, 200, 200))
            blank_photo = ImageTk.PhotoImage(blank)
            lbl.configure(image=blank_photo)
            lbl._blank_image = blank_photo  # Keep reference
            self.thumbnails.append(lbl)
            self.thumbnail_images.append(None)

    # --- Log panel ---
    def _build_log(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Log", padding=10)
        frame.grid(row=4, column=0, padx=10, pady=(5, 10), sticky="ew")
        frame.grid_columnconfigure(0, weight=1)

        # Progress bar + label at top of log section
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, columnspan=2, padx=5, pady=(5, 2), sticky="ew")

        self.progress_label = ttk.Label(frame, text="Idle")
        self.progress_label.grid(row=1, column=0, columnspan=2, padx=5, pady=(0, 5), sticky="w")

        # Log text box
        self.log_box = tk.Text(frame, height=10, wrap="word")
        self.log_box.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        # Add scrollbar for log
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.log_box.yview)
        scrollbar.grid(row=2, column=1, sticky="ns")
        self.log_box.configure(yscrollcommand=scrollbar.set)

    def _log(self, msg: str) -> None:
        """Append message to log box."""
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")

    # --- Control buttons ---
    def _build_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent, padding=10)
        frame.grid(row=3, column=0, padx=10, pady=(5, 10), sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        # Run / Cancel on left
        self.run_button = ttk.Button(
            frame, text="Start", command=self._on_run
        )
        self.run_button.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")

        self.cancel_button = ttk.Button(
            frame,
            text="Cancel",
            command=self._on_cancel,
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Open output folder button on right
        self.open_output_button = ttk.Button(
            frame,
            text="Open Output Folder",
            command=self._open_output_folder,
        )
        self.open_output_button.grid(row=0, column=2, padx=(5, 0), pady=5, sticky="e")

    def _open_output_folder(self) -> None:
        """Open output folder in system file explorer."""
        root = config.OUTPUT_ROOT
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["explorer", str(root)])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(root)])
            else:
                subprocess.Popen(["xdg-open", str(root)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder:\n{e}")

    # --- Output folder polling ---
    def _start_output_polling(self) -> None:
        """Start polling output folder for new images."""
        self.known_images = set()
        # Scan existing images so we don't show old ones
        try:
            for img_path in config.OUTPUT_ROOT.rglob("*_crop.png"):
                self.known_images.add(str(img_path))
            for img_path in config.OUTPUT_ROOT.rglob("*_overlay.png"):
                self.known_images.add(str(img_path))
        except Exception:
            pass

        self.polling_active = True
        self._poll_output_folder()

    def _stop_output_polling(self) -> None:
        """Stop polling."""
        self.polling_active = False

    def _start_elapsed_timer(self) -> None:
        """Start the elapsed time counter."""
        self.elapsed_timer_active = True
        self._tick_elapsed()

    def _stop_elapsed_timer(self) -> None:
        """Stop the elapsed time counter."""
        self.elapsed_timer_active = False

    def _tick_elapsed(self) -> None:
        """Update elapsed time display every second."""
        if not self.elapsed_timer_active:
            return

        self._update_progress_display()

        # Schedule next tick
        self.root.after(1000, self._tick_elapsed)

    def _poll_output_folder(self) -> None:
        """Check for new images in output folder."""
        if not self.polling_active:
            return

        try:
            # Check for new crop images
            new_images = []
            for img_path in config.OUTPUT_ROOT.rglob("*_crop.png"):
                path_str = str(img_path)
                if path_str not in self.known_images:
                    self.known_images.add(path_str)
                    new_images.append((img_path.stat().st_mtime, path_str))

            # Also check overlay images
            for img_path in config.OUTPUT_ROOT.rglob("*_overlay.png"):
                path_str = str(img_path)
                if path_str not in self.known_images:
                    self.known_images.add(path_str)
                    new_images.append((img_path.stat().st_mtime, path_str))

            # Sort by modification time and add newest ones
            new_images.sort(key=lambda x: x[0])
            for _, path_str in new_images:
                self._add_thumbnail(path_str)

            # Update progress based on crop count
            crop_count = len([p for p in self.known_images if "_crop.png" in p])
            if self.progress_total > 0 and crop_count > self.progress_current:
                # Update progress based on actual outputs
                self.progress_current = crop_count
                self._update_progress_display()

        except Exception:
            pass  # Ignore errors during polling

        # Schedule next poll
        if self.polling_active:
            self.root.after(500, self._poll_output_folder)  # Poll every 500ms

    def _add_thumbnail(self, path: str) -> None:
        """Add a new thumbnail to the display."""
        try:
            # Append new path, keep last 5
            self.thumbnail_paths.append(path)
            if len(self.thumbnail_paths) > 5:
                self.thumbnail_paths.pop(0)

            self._refresh_thumbnails()

        except Exception:
            pass  # Silently ignore thumbnail errors

    def _refresh_thumbnails(self) -> None:
        """Refresh all thumbnail displays."""
        # Update display (newest on right, oldest on left)
        n_paths = len(self.thumbnail_paths)

        for i in range(5):
            thumb = self.thumbnails[i]
            # Index from paths: leftmost shows oldest, rightmost shows newest
            path_idx = i - (5 - n_paths)

            if path_idx >= 0 and path_idx < n_paths:
                img_path = self.thumbnail_paths[path_idx]

                try:
                    img = Image.open(img_path)
                    img.thumbnail((110, 110))

                    # Create PhotoImage and store reference
                    photo = ImageTk.PhotoImage(img)
                    self.thumbnail_images[i] = photo

                    thumb.configure(image=photo)

                    # Bind click handler - use default argument to capture value
                    thumb.unbind("<Button-1>")
                    thumb.bind("<Button-1>", lambda e, p=img_path: self._open_image(p))

                except Exception:
                    # Silently handle image loading errors
                    try:
                        thumb.configure(image=thumb._blank_image)
                        thumb.unbind("<Button-1>")
                    except Exception:
                        pass
            else:
                try:
                    thumb.configure(image=thumb._blank_image)
                    self.thumbnail_images[i] = None
                    thumb.unbind("<Button-1>")
                except Exception:
                    pass

    def _update_progress_display(self) -> None:
        """Update progress bar and label with elapsed/ETA."""
        current = self.progress_current
        total = self.progress_total

        if total > 0:
            self.progress_var.set((current / total) * 100)
        else:
            self.progress_var.set(0)

        elapsed_text = ""
        eta_text = ""

        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            elapsed_text = f" | Elapsed: {format_time(elapsed)}"

            if current > 0 and total > 0:
                avg_per_item = elapsed / current
                remaining = max(0, total - current)
                eta_seconds = avg_per_item * remaining
                eta_text = f" | ETA: {format_time(eta_seconds)}"

        self.progress_label.configure(
            text=f"Processing: {current}/{total}{elapsed_text}{eta_text}"
        )

    # --- Worker thread ---
    def _get_config(self) -> Dict[str, Any]:
        """Collect config from widgets."""
        config: Dict[str, Any] = {}

        config["quick_test"] = self.mode_var.get() == "quick"
        config["safe_mode"] = self.safe_var.get()

        if not config["quick_test"]:
            # Sampling
            sample = self.sample_var.get()
            if sample == "custom":
                try:
                    config["step"] = int(self.custom_step.get())
                except ValueError:
                    config["step"] = 10
            else:
                config["step"] = int(sample)

            # Calibration
            config["global_mode"] = self.calib_var.get() == "global"

            # Outputs
            config["full_output"] = self.output_var.get() == "all"

            # Profiling
            config["profile"] = self.profile_var.get()

            # Focus classification
            config["focus_classification"] = self.focus_class_var.get()
        else:
            # Defaults that won't actually be used in quick mode
            config["step"] = 10
            config["global_mode"] = True
            config["full_output"] = False
            config["profile"] = False
            config["focus_classification"] = False

        return config

    def _estimate_total_units(self, step: int) -> int:
        """Estimate total units for progress (approx droplets * 2 cameras)."""
        try:
            root = self.selected_root or config.CINE_ROOT
            from cine_io import get_cine_folders, group_cines_by_droplet

            cine_folders = get_cine_folders(root)
            droplets = 0
            for folder in cine_folders:
                droplets += len(group_cines_by_droplet(folder))

            # Each droplet has ~2 cameras, step reduces count
            return max(1, (droplets * 2) // max(1, step))
        except Exception:
            return 1

    def _on_run(self) -> None:
        """Start processing."""
        if self.processing:
            return

        if self.selected_root is None:
            messagebox.showerror("No folder selected", "Please select a cine folder first.")
            self._show_landing()
            return

        self.processing = True
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")

        # Clear state
        self.log_box.delete("1.0", "end")
        self.progress_var.set(0)
        self.progress_label.configure(text="Processing...")
        self.thumbnail_paths.clear()
        for i, thumb in enumerate(self.thumbnails):
            thumb.configure(image=thumb._blank_image)
            self.thumbnail_images[i] = None
            thumb.unbind("<Button-1>")

        # Get config
        config = self._get_config()
        self._log(f"Starting with config: {config}")

        # Reset timers + progress
        self.start_time = time.time()
        self.progress_current = 0
        self.progress_total = 0

        if not config["quick_test"]:
            self.progress_total = self._estimate_total_units(config["step"])
        else:
            # For quick test, total is number of folders
            try:
                from cine_io import get_cine_folders
                root = self.selected_root or config.CINE_ROOT
                self.progress_total = len(get_cine_folders(root))
            except Exception:
                self.progress_total = 1

        # Start output folder polling for thumbnails
        self._start_output_polling()

        # Start elapsed timer
        self._start_elapsed_timer()

        # Start worker thread
        self.worker_thread = threading.Thread(
            target=self._run_worker,
            args=(config,),
            daemon=True,
        )
        self.worker_thread.start()

    def _run_worker(self, config: Dict[str, Any]) -> None:
        """Run processing in background thread."""
        try:
            if config["quick_test"]:
                self._run_quick_test(config["safe_mode"])
            else:
                self._run_full_pipeline(config)
        except Exception as e:
            emit_log(f"ERROR: {e}")
        finally:
            emit_done()

    def _run_quick_test(self, safe_mode: bool) -> None:
        """Run quick detection test (1st droplet per folder)."""
        from cine_io import (
            group_cines_by_droplet,
            get_cine_folders,
            safe_load_cine,
        )
        from darkness_analysis import (
            analyze_cine_darkness,
            choose_best_frame_with_geo,
        )
        from image_utils import otsu_mask
        from plotting import save_darkness_plot, save_geometric_overlay

        root = self.selected_root or config.CINE_ROOT
        cine_folders = get_cine_folders(root)
        total = len(cine_folders)

        if total == 0:
            emit_log("No folders with .cine files found!")
            return

        emit_log(f"Quick detection test: {total} folder{'s' if total != 1 else ''}")

        for idx, sub in enumerate(cine_folders, start=1):
            emit_progress(idx, total)
            emit_log(f"[{idx}/{total}] {sub.name}")

            groups = group_cines_by_droplet(sub)
            if not groups:
                emit_log("  No cines, skipping")
                continue

            droplet_id, cams = groups[0]
            path = cams.get("g") or cams.get("v")
            if path is None:
                continue

            cine_obj = safe_load_cine(path)
            if cine_obj is None:
                continue

            first, last = cine_obj.range

            dark = analyze_cine_darkness(cine_obj)
            curve = dark["darkness_curve"]

            best_idx, geo = choose_best_frame_with_geo(cine_obj, curve)

            out_sub = config.OUTPUT_ROOT / sub.name
            out_sub.mkdir(parents=True, exist_ok=True)

            save_darkness_plot(
                out_sub / f"{path.stem}_darkness.png",
                curve,
                first,
                last,
                best_idx,
                path.name,
            )

            frame = geo["frame"]
            _, mask = otsu_mask(frame)
            geo_for_plot = {
                "frame": frame,
                "mask": mask,
                "y_top": geo["y_top"],
                "y_bottom": geo["y_bottom"],
                "y_bottom_sphere": geo["y_bottom_sphere"],
                "cx": geo["cx"],
            }

            overlay_path = out_sub / f"{path.stem}_overlay.png"
            save_geometric_overlay(overlay_path, geo_for_plot, best_idx, cnn_size=None)

            emit_log(f"  Best frame: {best_idx}")

        emit_log("Quick detection test complete!")

    def _run_full_pipeline(self, run_config: Dict[str, Any]) -> None:
        """Run full pipeline (global or per-folder) with GUI logs."""
        from pipeline_folder import process_per_folder
        from pipeline_global import process_global

        # Apply step to config module
        config.CINE_STEP = run_config["step"]

        emit_log(
            f"Cine step: {run_config['step']}, "
            f"Mode: {'Global' if run_config['global_mode'] else 'Per-folder'}, "
            f"Outputs: {'All plots' if run_config['full_output'] else 'Crops only'}"
        )

        # --- Mirror all prints to GUI log as well ---
        original_print = builtins.print

        def gui_print(*args: Any, **kwargs: Any) -> None:
            text = " ".join(str(a) for a in args)

            # Check for progress marker
            if text.startswith("__PROGRESS__:"):
                # Parse: __PROGRESS__:current:total:desc
                parts = text.split(":", 3)
                if len(parts) == 4:
                    gui_queue.put(("increment", parts[3]))
                return  # Don't print to CLI or log

            # Normal CLI output
            original_print(*args, **kwargs)
            # Mirror into GUI log
            try:
                emit_log(text)
            except Exception:
                pass

        builtins.print = gui_print

        try:
            if run_config["global_mode"]:
                process_global(
                    safe_mode=run_config["safe_mode"],
                    profile=run_config["profile"],
                    quick_test=False,
                    full_output=run_config["full_output"],
                    gui_mode=True,
                    focus_classification=run_config["focus_classification"],
                )
            else:
                process_per_folder(
                    safe_mode=run_config["safe_mode"],
                    profile=run_config["profile"],
                    quick_test=False,
                    full_output=run_config["full_output"],
                    gui_mode=True,
                    focus_classification=run_config["focus_classification"],
                )
        finally:
            # Restore print
            builtins.print = original_print

        emit_log("Pipeline complete!")

    def _on_cancel(self) -> None:
        """Cancel processing (best effort)."""
        self._log("Cancelling... (will stop after current item)")
        # Real cooperative cancellation would need pipeline hooks
        self.processing = False
        self._stop_output_polling()
        self._stop_elapsed_timer()

    def _on_done(self) -> None:
        """Handle processing complete."""
        self.processing = False
        self._stop_output_polling()
        self._stop_elapsed_timer()
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.progress_label.configure(text="Complete!")
        self.progress_var.set(100)

    # --- Queue polling ---
    def _open_image(self, path: str) -> None:
        """Open a single image in the system viewer."""
        try:
            p = Path(path)
            if not p.exists():
                self._log(f"Image not found: {p}")
                return

            if platform.system() == "Windows":
                subprocess.Popen(["explorer", str(p)])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            self._log(f"Error opening image: {e}")

    def _poll_queue(self) -> None:
        """Poll queue for worker messages (logs, progress, done)."""
        try:
            while True:
                msg_type, data = gui_queue.get_nowait()

                if msg_type == "log":
                    self._log(data)

                elif msg_type == "increment":
                    # Increment global progress counter
                    self.progress_current += 1
                    self._update_progress_display()
                    self.root.update_idletasks()  # Force UI refresh

                elif msg_type == "progress":
                    # Legacy format (current, total) or (current, total, desc)
                    if len(data) == 3:
                        current, total, desc = data
                    else:
                        current, total = data
                    self.progress_current = current
                    self.progress_total = total
                    self._update_progress_display()

                elif msg_type == "done":
                    self._on_done()

        except queue.Empty:
            pass

        # Schedule next poll
        self.root.after(100, self._poll_queue)



def run_gui() -> None:
    """Launch the GUI."""
    app = PipelineGUI()
    app.mainloop()


if __name__ == "__main__":
    run_gui()
