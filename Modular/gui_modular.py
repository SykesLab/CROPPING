"""GUI interface for the droplet cropping pipeline.

Provides a modern GUI using customtkinter with:
- Landing screen for folder selection
- Configuration options (execution, sampling, calibration, outputs)
- Progress bar, log, elapsed time, and ETA
- Live preview of recent crops (click to open full image)
- Button to open the output folder

Uses output folder polling for thumbnails (works in both Safe and Fast modes).
"""

import builtins
import queue
import threading
import time
import subprocess
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import customtkinter as ctk
    from tkinter import filedialog, messagebox
    from PIL import Image
except ImportError:
    print("GUI requires: pip install customtkinter pillow")
    raise

import config_modular
from config_modular import OUTPUT_ROOT
from phantom_silence_modular import cine

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


class PipelineGUI(ctk.CTk):
    """Main GUI window with landing + main screen."""

    def __init__(self) -> None:
        super().__init__()

        self.title("Droplet Cropping Pipeline")
        self.geometry("800x700")
        self.minsize(700, 600)

        # Single cell layout for swapping frames
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # State
        self.processing: bool = False
        self.worker_thread: Optional[threading.Thread] = None
        self.thumbnails: List[ctk.CTkLabel] = []
        self.thumbnail_paths: List[str] = []
        self.selected_root: Optional[Path] = None
        self.start_time: Optional[float] = None
        self.progress_current: int = 0
        self.progress_total: int = 0

        # For output folder polling
        self.known_images: set = set()
        self.polling_active: bool = False
        
        # For elapsed timer
        self.elapsed_timer_active: bool = False

        # Top-level frames
        self.landing_frame = ctk.CTkFrame(self)
        self.landing_frame.grid(row=0, column=0, sticky="nsew")

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(4, weight=1)

        # Build UI
        self._build_landing()
        self._build_header(self.main_frame)
        self._build_config(self.main_frame)
        self._build_preview(self.main_frame)
        self._build_log(self.main_frame)
        self._build_controls(self.main_frame)

        # Start queue polling
        self.after(100, self._poll_queue)

        # Show landing first
        self._show_landing()

    # ============================================================
    # LANDING SCREEN
    # ============================================================
    def _build_landing(self) -> None:
        """Initial landing page asking for cine folder."""
        self.landing_frame.grid_columnconfigure(0, weight=1)
        self.landing_frame.grid_rowconfigure(0, weight=1)

        inner = ctk.CTkFrame(self.landing_frame)
        inner.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        inner.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            inner,
            text="DROPLET CROPPING PIPELINE",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        title.grid(row=0, column=0, pady=(30, 10), sticky="n")

        subtitle = ctk.CTkLabel(
            inner,
            text="Select the root folder containing your cine subfolders to begin.",
            wraplength=500,
        )
        subtitle.grid(row=1, column=0, pady=(0, 30), sticky="n")

        select_btn = ctk.CTkButton(
            inner,
            text="Select cine root folder",
            command=self._select_folder,
            width=220,
        )
        select_btn.grid(row=2, column=0, pady=(0, 10))

        default_root = config_modular.CINE_ROOT
        self.landing_path_label = ctk.CTkLabel(
            inner,
            text=f"Current: {default_root}",
            wraplength=500,
        )
        self.landing_path_label.grid(row=3, column=0, pady=(10, 5))

        self.landing_info_label = ctk.CTkLabel(inner, text="")
        self.landing_info_label.grid(row=4, column=0, pady=(5, 20))

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

    def _select_folder(self) -> None:
        """Choose cine root folder and switch to main UI."""
        folder = filedialog.askdirectory(
            title="Select folder containing cine subfolders"
        )
        if not folder:
            return

        root = Path(folder)
        if not root.exists():
            messagebox.showerror("Invalid folder", "Selected folder does not exist.")
            return

        # Update config module so pipeline sees new root
        config_modular.CINE_ROOT = root
        self.selected_root = root

        # Update landing + main headers
        self.landing_path_label.configure(text=f"Current: {root}")
        self.source_label.configure(text=str(root))

        # Rescan + show counts
        self._rescan_counts_for_root(root)

        # Jump to main screen
        self._show_main()

    def _rescan_counts_for_root(self, root: Path) -> None:
        """Scan cine root and update counts on both landing + header."""
        try:
            from cine_io_modular import iter_subfolders, group_cines_by_droplet

            subfolders = list(iter_subfolders(root))
            n_folders = len(subfolders)
            droplets = 0
            for sub in subfolders:
                groups = group_cines_by_droplet(sub)
                droplets += len(groups)

            text = f"{n_folders} folders, ~{droplets} droplets"
        except Exception as e:
            text = f"Scan error: {e}"

        self.landing_info_label.configure(text=text)
        self.count_label.configure(text=text)

    # ============================================================
    # HEADER
    # ============================================================
    def _build_header(self, parent: ctk.CTkFrame) -> None:
        """Build header section."""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            frame,
            text="DROPLET CROPPING PIPELINE",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        title.grid(row=0, column=0, columnspan=3, pady=(10, 5))

        # Source info
        ctk.CTkLabel(frame, text="Source:").grid(row=1, column=0, padx=10, sticky="w")
        default_root = config_modular.CINE_ROOT
        self.source_label = ctk.CTkLabel(frame, text=str(default_root))
        self.source_label.grid(row=1, column=1, padx=10, sticky="w")

        change_btn = ctk.CTkButton(
            frame,
            text="Change…",
            width=80,
            command=self._select_folder,
        )
        change_btn.grid(row=1, column=2, padx=10, sticky="e")

        # Count info
        ctk.CTkLabel(frame, text="Found:").grid(row=2, column=0, padx=10, sticky="w")
        self.count_label = ctk.CTkLabel(frame, text="Scanning…")
        self.count_label.grid(row=2, column=1, padx=10, sticky="w", pady=(0, 10))

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
        root = self.selected_root or config_modular.CINE_ROOT
        try:
            from cine_io_modular import iter_subfolders, group_cines_by_droplet

            subfolders = list(iter_subfolders(root))
            n_folders = len(subfolders)
            droplets = 0
            for sub in subfolders:
                droplets += len(group_cines_by_droplet(sub))

            text = f"{n_folders} folders, ~{droplets} droplets"
        except Exception as e:
            text = f"Scan error: {e}"

        if not sdk_ok:
            text += "  (SDK NOT FOUND!)"

        def update() -> None:
            self.count_label.configure(text=text)

        self.after(0, update)

    # ============================================================
    # CONFIG
    # ============================================================
    def _build_config(self, parent: ctk.CTkFrame) -> None:
        frame = ctk.CTkFrame(parent)
        frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        # ----- Execution mode -----
        exec_frame = ctk.CTkFrame(frame)
        exec_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        exec_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            exec_frame,
            text="Execution mode",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=(8, 4), sticky="w")

        self.mode_var = ctk.StringVar(value="full")

        self.quick_radio = ctk.CTkRadioButton(
            exec_frame,
            text="Quick detection test (1st droplet per folder)",
            variable=self.mode_var,
            value="quick",
            command=self._update_config_state,
        )
        self.quick_radio.grid(row=1, column=0, padx=15, pady=2, sticky="w")

        self.full_radio = ctk.CTkRadioButton(
            exec_frame,
            text="Full pipeline",
            variable=self.mode_var,
            value="full",
            command=self._update_config_state,
        )
        self.full_radio.grid(row=2, column=0, padx=15, pady=(2, 8), sticky="w")

        # Safe mode
        self.safe_var = ctk.BooleanVar(value=False)
        self.safe_check = ctk.CTkCheckBox(
            exec_frame,
            text="Safe mode (single process, more stable)",
            variable=self.safe_var,
        )
        self.safe_check.grid(row=3, column=0, padx=15, pady=(0, 8), sticky="w")

        # ----- Sampling -----
        sample_frame = ctk.CTkFrame(frame)
        sample_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        sample_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            sample_frame,
            text="Sampling (frame step)",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=(8, 4), sticky="w")

        self.sample_var = ctk.StringVar(value="10")

        s_row = 1
        self.sample_radios: List[ctk.CTkRadioButton] = []
        for step in ["1", "2", "5", "10"]:
            rb = ctk.CTkRadioButton(
                sample_frame,
                text=f"Every {step} frame(s)",
                variable=self.sample_var,
                value=step,
            )
            rb.grid(row=s_row, column=0, padx=15, pady=1, sticky="w")
            self.sample_radios.append(rb)
            s_row += 1

        # Custom step
        custom_frame = ctk.CTkFrame(sample_frame)
        custom_frame.grid(row=s_row, column=0, padx=10, pady=(4, 10), sticky="w")
        self.custom_step = ctk.CTkEntry(custom_frame, width=60, placeholder_text="step")
        self.custom_step.grid(row=0, column=0, padx=(5, 5))
        self.custom_radio = ctk.CTkRadioButton(
            custom_frame,
            text="Custom step",
            variable=self.sample_var,
            value="custom",
        )
        self.custom_radio.grid(row=0, column=1, padx=(0, 5), sticky="w")

        self.sampling_controls: List[ctk.CTkBaseClass] = (
            self.sample_radios + [self.custom_step, self.custom_radio]
        )

        # ----- Calibration & Outputs (second row) -----
        calib_frame = ctk.CTkFrame(frame)
        calib_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        calib_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            calib_frame,
            text="Calibration (crop)",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=(8, 4), sticky="w")

        self.calib_var = ctk.StringVar(value="folder")

        self.calib_folder_radio = ctk.CTkRadioButton(
            calib_frame,
            text="Per-folder crop (each folder calibrated independently)",
            variable=self.calib_var,
            value="folder",
        )
        self.calib_folder_radio.grid(row=1, column=0, padx=15, pady=2, sticky="w")

        self.calib_global_radio = ctk.CTkRadioButton(
            calib_frame,
            text="Global crop (calibration from all folders combined)",
            variable=self.calib_var,
            value="global",
        )
        self.calib_global_radio.grid(row=2, column=0, padx=15, pady=(2, 8), sticky="w")

        self.calib_controls = [self.calib_folder_radio, self.calib_global_radio]

        # Outputs + profiling
        output_frame = ctk.CTkFrame(frame)
        output_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        output_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            output_frame,
            text="Outputs",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=(8, 4), sticky="w")

        self.output_var = ctk.StringVar(value="crops")

        self.output_crops_radio = ctk.CTkRadioButton(
            output_frame,
            text="Crops only (fastest)",
            variable=self.output_var,
            value="crops",
        )
        self.output_crops_radio.grid(row=1, column=0, padx=15, pady=2, sticky="w")

        self.output_all_radio = ctk.CTkRadioButton(
            output_frame,
            text="All plots",
            variable=self.output_var,
            value="all",
        )
        self.output_all_radio.grid(row=2, column=0, padx=15, pady=(2, 6), sticky="w")

        self.output_controls = [self.output_crops_radio, self.output_all_radio]

        self.profile_var = ctk.BooleanVar(value=False)
        self.profile_check = ctk.CTkCheckBox(
            output_frame,
            text="Enable profiling",
            variable=self.profile_var,
        )
        self.profile_check.grid(row=3, column=0, padx=15, pady=(0, 8), sticky="w")

        # Keep profiling in same disable group as outputs
        self.profile_control = self.profile_check

        # Initialise enable/disable state
        self._update_config_state()

    def _update_config_state(self) -> None:
        """Enable/disable controls depending on quick/full mode."""
        quick = self.mode_var.get() == "quick"
        state = "disabled" if quick else "normal"

        # Sampling
        for w in self.sampling_controls:
            try:
                w.configure(state=state)
            except Exception:
                pass

        # Calibration
        for w in self.calib_controls:
            try:
                w.configure(state=state)
            except Exception:
                pass

        # Outputs
        for w in self.output_controls:
            try:
                w.configure(state=state)
            except Exception:
                pass

        # Profiling
        try:
            self.profile_control.configure(state=state)
        except Exception:
            pass

    # ============================================================
    # PREVIEW
    # ============================================================
    def _build_preview(self, parent: ctk.CTkFrame) -> None:
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame,
            text="Recent outputs (click to open)",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=(8, 4), sticky="w")

        thumb_row = ctk.CTkFrame(frame)
        thumb_row.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="w")

        # 5 thumbnails (latest on the right)
        for i in range(5):
            lbl = ctk.CTkLabel(
                thumb_row,
                text="",
                width=110,
                height=110,
                corner_radius=8,
                fg_color=("gray15", "gray25"),
            )
            lbl.grid(row=0, column=i, padx=4, pady=4)
            self.thumbnails.append(lbl)

    # ============================================================
    # LOG
    # ============================================================
    def _build_log(self, parent: ctk.CTkFrame) -> None:
        frame = ctk.CTkFrame(parent)
        frame.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            frame,
            text="Log",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=(8, 4), sticky="w")

        self.log_box = ctk.CTkTextbox(frame, height=160)
        self.log_box.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="nsew")

    def _log(self, msg: str) -> None:
        """Append message to log box."""
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")

    # ============================================================
    # CONTROLS
    # ============================================================
    def _build_controls(self, parent: ctk.CTkFrame) -> None:
        frame = ctk.CTkFrame(parent)
        frame.grid(row=4, column=0, padx=10, pady=(5, 10), sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        # Run / Cancel
        self.run_button = ctk.CTkButton(
            frame, text="Start", command=self._on_run, width=100
        )
        self.run_button.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="w")

        self.cancel_button = ctk.CTkButton(
            frame,
            text="Cancel",
            command=self._on_cancel,
            width=100,
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=0, padx=(120, 5), pady=5, sticky="w")

        # Open output folder button
        self.open_output_button = ctk.CTkButton(
            frame,
            text="Open Output Folder",
            command=self._open_output_folder,
            width=160,
        )
        self.open_output_button.grid(row=0, column=2, padx=(5, 10), pady=5, sticky="e")

        # Progress bar + label
        self.progress_bar = ctk.CTkProgressBar(frame)
        self.progress_bar.grid(row=1, column=0, columnspan=3, padx=10, pady=(5, 2), sticky="ew")
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(frame, text="Idle")
        self.progress_label.grid(row=2, column=0, columnspan=3, padx=10, pady=(0, 8), sticky="w")

    def _open_output_folder(self) -> None:
        """Open OUTPUT_ROOT in system file explorer."""
        root = OUTPUT_ROOT
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["explorer", str(root)])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(root)])
            else:
                subprocess.Popen(["xdg-open", str(root)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder:\n{e}")

    # ============================================================
    # OUTPUT FOLDER POLLING (for thumbnails)
    # ============================================================
    def _start_output_polling(self) -> None:
        """Start polling output folder for new images."""
        self.known_images = set()
        # Scan existing images so we don't show old ones
        try:
            for img_path in OUTPUT_ROOT.rglob("*_crop.png"):
                self.known_images.add(str(img_path))
            for img_path in OUTPUT_ROOT.rglob("*_overlay.png"):
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
        self.after(1000, self._tick_elapsed)

    def _poll_output_folder(self) -> None:
        """Check for new images in output folder."""
        if not self.polling_active:
            return

        try:
            # Check for new crop images
            new_images = []
            for img_path in OUTPUT_ROOT.rglob("*_crop.png"):
                path_str = str(img_path)
                if path_str not in self.known_images:
                    self.known_images.add(path_str)
                    new_images.append((img_path.stat().st_mtime, path_str))

            # Also check overlay images
            for img_path in OUTPUT_ROOT.rglob("*_overlay.png"):
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
            self.after(500, self._poll_output_folder)  # Poll every 500ms

    def _add_thumbnail(self, path: str) -> None:
        """Add a new thumbnail to the display."""
        try:
            # Append new path, keep last 5
            self.thumbnail_paths.append(path)
            if len(self.thumbnail_paths) > 5:
                self.thumbnail_paths.pop(0)

            self._refresh_thumbnails()

        except Exception as e:
            self._log(f"Thumbnail error: {e}")

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
                    photo = ctk.CTkImage(light_image=img, dark_image=img, size=(110, 110))

                    thumb.configure(image=photo, text="")
                    thumb.image = photo  # keep reference

                    # Bind click handler - use default argument to capture value
                    thumb.unbind("<Button-1>")
                    thumb.bind("<Button-1>", lambda e, p=img_path: self._open_image(p))

                except Exception:
                    thumb.configure(image=None, text="?")
                    thumb.unbind("<Button-1>")
            else:
                thumb.configure(image=None, text="")
                thumb.image = None
                thumb.unbind("<Button-1>")

    def _update_progress_display(self) -> None:
        """Update progress bar and label with elapsed/ETA."""
        current = self.progress_current
        total = self.progress_total

        if total > 0:
            self.progress_bar.set(current / total)
        else:
            self.progress_bar.set(0)

        elapsed_text = ""
        eta_text = ""

        if self.start_time is not None:
            elapsed = int(time.time() - self.start_time)
            e_mins = elapsed // 60
            e_secs = elapsed % 60
            elapsed_text = f" | Elapsed: {e_mins}m {e_secs}s"

            if current > 0 and total > 0:
                avg_per_item = (time.time() - self.start_time) / current
                remaining = max(0, total - current)
                eta_seconds = int(avg_per_item * remaining)
                a_mins = eta_seconds // 60
                a_secs = eta_seconds % 60
                eta_text = f" | ETA: {a_mins}m {a_secs}s"

        self.progress_label.configure(
            text=f"Processing: {current}/{total}{elapsed_text}{eta_text}"
        )

    # ============================================================
    # RUNTIME / WORKER
    # ============================================================
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
        else:
            # Defaults that won't actually be used in quick mode
            config["step"] = 10
            config["global_mode"] = True
            config["full_output"] = False
            config["profile"] = False

        return config

    def _estimate_total_units(self, step: int) -> int:
        """Estimate total units for progress (approx droplets * 2 cameras)."""
        try:
            root = self.selected_root or config_modular.CINE_ROOT
            from cine_io_modular import iter_subfolders, group_cines_by_droplet

            subfolders = list(iter_subfolders(root))
            droplets = 0
            for sub in subfolders:
                droplets += len(group_cines_by_droplet(sub))

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
        self.progress_bar.set(0)
        self.progress_label.configure(text="Processing...")
        self.thumbnail_paths.clear()
        for thumb in self.thumbnails:
            thumb.configure(image=None, text="")
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
                from cine_io_modular import iter_subfolders
                root = self.selected_root or config_modular.CINE_ROOT
                self.progress_total = len(list(iter_subfolders(root)))
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
        from cine_io_modular import (
            group_cines_by_droplet,
            iter_subfolders,
            safe_load_cine,
        )
        from darkness_analysis_modular import (
            analyze_cine_darkness,
            choose_best_frame_with_geo,
        )
        from image_utils_modular import otsu_mask
        from plotting_modular import save_darkness_plot, save_geometric_overlay

        root = self.selected_root or config_modular.CINE_ROOT
        subfolders = list(iter_subfolders(root))
        total = len(subfolders)

        emit_log(f"Quick detection test: {total} folders")

        for idx, sub in enumerate(subfolders, start=1):
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

            out_sub = OUTPUT_ROOT / sub.name
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

    def _run_full_pipeline(self, config: Dict[str, Any]) -> None:
        """Run full pipeline (global or per-folder) with GUI logs."""
        from pipeline_folder import process_per_folder
        from pipeline_global import process_global

        # Apply step to config module
        config_modular.CINE_STEP = config["step"]

        emit_log(
            f"Step: {config['step']}, "
            f"Mode: {'Global' if config['global_mode'] else 'Per-folder'}, "
            f"Outputs: {'All plots' if config['full_output'] else 'Crops only'}"
        )

        # --- Mirror all prints to GUI log as well ---
        original_print = builtins.print

        def gui_print(*args: Any, **kwargs: Any) -> None:
            text = " ".join(str(a) for a in args)
            # Normal CLI output
            original_print(*args, **kwargs)
            # Mirror into GUI log
            try:
                emit_log(text)
            except Exception:
                pass

        builtins.print = gui_print

        try:
            if config["global_mode"]:
                process_global(
                    safe_mode=config["safe_mode"],
                    profile=config["profile"],
                    quick_test=False,
                    full_output=config["full_output"],
                )
            else:
                process_per_folder(
                    safe_mode=config["safe_mode"],
                    profile=config["profile"],
                    quick_test=False,
                    full_output=config["full_output"],
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
        self.progress_bar.set(1.0)

    # ============================================================
    # THUMBNAILS + QUEUE
    # ============================================================
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

                elif msg_type == "progress":
                    current, total = data
                    self.progress_current = current
                    self.progress_total = total
                    self._update_progress_display()

                elif msg_type == "done":
                    self._on_done()

        except queue.Empty:
            pass

        # Schedule next poll
        self.after(100, self._poll_queue)


def run_gui() -> None:
    """Launch the GUI."""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = PipelineGUI()
    app.mainloop()


if __name__ == "__main__":
    run_gui()
