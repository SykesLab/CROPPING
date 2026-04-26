"""Tier-A multi-tab Tk GUI for the fingerprint checker.

Layout:
    Top    — folder pickers (config / synthetic / calibration / output)
              + "n_synth_samples" entry + Run All button
    Middle — Notebook with tabs:
              Overview / Check A / Sigma Trends / Check B / Report
    Bottom — status bar + progress bar

Worker thread runs the orchestrator so the UI stays responsive. Plots
are embedded via FigureCanvasTkAgg.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np
import pandas as pd

# Path setup so tools/fingerprint_checker is importable when run standalone
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _module in ("Calibration", "Training", ""):
    _p = str(_REPO_ROOT / _module) if _module else str(_REPO_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use('Agg')
from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)

from tools.fingerprint_checker.fingerprint_analyses import (  # noqa: E402
    find_nearest_match,
)
from tools.fingerprint_checker.fingerprint_cache import (  # noqa: E402
    DEFAULT_CACHE_DIR, clear_cache,
)
from tools.fingerprint_checker.fingerprint_io import (  # noqa: E402
    load_sample_image_by_row,
)
from tools.fingerprint_checker.fingerprint_orchestrator import (  # noqa: E402
    AllChecksResult, run_all_checks,
)
from tools.fingerprint_checker.fingerprint_plots import (  # noqa: E402
    plot_alignment_per_anchor_heatmap,
    plot_alignment_per_metric_deltas,
    plot_coverage_bars,
    plot_scale_chain_residuals,
    plot_sigma_trends,
)
from tools.fingerprint_checker.fingerprint_report import write_full_report  # noqa: E402


class FingerprintCheckerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Blur / Domain Fingerprint Checker")
        self.geometry("1100x800")

        # Default paths (try to be helpful)
        default_root = _REPO_ROOT / "Training" / "training_output"
        default_models = default_root / "models"
        default_dataset_root = default_root / "datasets"

        # State
        self.config_var = tk.StringVar(value=self._first_existing(
            default_models, "*.pth", "checkpoints/dme_best.pth", suffix=True))
        self.synth_var = tk.StringVar(value=self._latest_dataset(default_dataset_root))
        self.calib_var = tk.StringVar(value="")
        self.inference_var = tk.StringVar(value="")
        self.output_var = tk.StringVar(value=str(_REPO_ROOT / "tools" / "fingerprint_checker" / "output" / datetime.now().strftime("%Y%m%d_%H%M%S")))
        self.n_synth_var = tk.StringVar(value=str(self._sample_default(self.synth_var.get())))
        self.n_inference_var = tk.StringVar(value="")
        self.k_var = tk.StringVar(value="20")
        self.focus_offset_var = tk.StringVar(value="0.0")
        self.force_recompute_var = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="Ready. Pick a config + synthetic dataset (and optionally calibration), then Run All.")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._result: Optional[AllChecksResult] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._msg_queue: "queue.Queue" = queue.Queue()
        self._embedded_canvases = []  # (FigureCanvasTkAgg, frame) list for cleanup
        self._run_start_time: Optional[float] = None

        self._build_ui()
        self.after(100, self._poll_messages)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _first_existing(parent: Path, *_args, **_kwargs) -> str:
        if not parent.is_dir():
            return ""
        for sub in sorted(parent.iterdir(), reverse=True):
            cfg = sub / "training_config.yaml"
            if cfg.is_file():
                return str(cfg)
        return ""

    @staticmethod
    def _latest_dataset(datasets_dir: Path) -> str:
        if not datasets_dir.is_dir():
            return ""
        ds = sorted([p for p in datasets_dir.iterdir()
                     if p.is_dir() and (p / "metadata.csv").is_file()],
                    reverse=True)
        return str(ds[0]) if ds else ""

    @staticmethod
    def _dataset_count(synth_path: str) -> int:
        if not synth_path:
            return 0
        meta = Path(synth_path) / "metadata.csv"
        if not meta.is_file():
            return 0
        try:
            with open(meta) as f:
                return sum(1 for _ in f) - 1  # subtract header
        except Exception:
            return 0

    @staticmethod
    def _sample_default(synth_path: str) -> int:
        """Default n_synth: full count when small, capped at 200 for large datasets."""
        count = FingerprintCheckerApp._dataset_count(synth_path)
        if count > 1000:
            return 200
        return count

    def _refresh_n_default(self, *_):
        self.n_synth_var.set(str(self._sample_default(self.synth_var.get())))

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        top = ttk.LabelFrame(self, text="Inputs", padding=8)
        top.pack(fill='x', padx=8, pady=(8, 4))

        def _row(label, var, browse_fn, width=70):
            r = ttk.Frame(top)
            r.pack(fill='x', pady=2)
            ttk.Label(r, text=label, width=22).pack(side='left')
            ttk.Entry(r, textvariable=var, width=width).pack(side='left', padx=4)
            ttk.Button(r, text="Browse", command=browse_fn).pack(side='left')
            return r

        _row("Config (yaml or .pth):", self.config_var, self._browse_config)
        _row("Synthetic dataset:", self.synth_var, self._browse_synth)
        _row("Calibration folder (opt):", self.calib_var, self._browse_calib)
        _row("Inference crops folder (opt):", self.inference_var, self._browse_inference)
        _row("Output folder:", self.output_var, self._browse_output)

        opts = ttk.Frame(top)
        opts.pack(fill='x', pady=(4, 2))
        ttk.Label(opts, text="Samples to fingerprint:  synth").pack(side='left', padx=(0, 2))
        ttk.Entry(opts, textvariable=self.n_synth_var, width=8).pack(side='left')
        ttk.Label(opts, text="  inference").pack(side='left', padx=(8, 2))
        ttk.Entry(opts, textvariable=self.n_inference_var, width=6).pack(side='left')
        ttk.Label(opts, text=" (blank = all)", foreground='gray',
                  font=('TkDefaultFont', 8)).pack(side='left', padx=4)

        opts2 = ttk.Frame(top)
        opts2.pack(fill='x', pady=2)
        ttk.Label(opts2, text="K nearest synthetic per anchor:").pack(side='left', padx=(0, 4))
        ttk.Entry(opts2, textvariable=self.k_var, width=6).pack(side='left')
        ttk.Label(opts2, text="    Calibration focus offset (mm):").pack(side='left', padx=(12, 4))
        ttk.Entry(opts2, textvariable=self.focus_offset_var, width=8).pack(side='left')

        btns = ttk.Frame(top)
        btns.pack(fill='x', pady=(6, 0))
        self.run_button = ttk.Button(btns, text="Run All Checks", command=self._on_run)
        self.run_button.pack(side='left', padx=2)
        self.save_button = ttk.Button(
            btns, text="Save Report", command=self._on_save, state='disabled')
        self.save_button.pack(side='left', padx=2)
        self.open_output_button = ttk.Button(
            btns, text="Open Output Folder", command=self._on_open_output)
        self.open_output_button.pack(side='left', padx=2)
        ttk.Separator(btns, orient='vertical').pack(
            side='left', fill='y', padx=8)
        ttk.Checkbutton(btns, text="Force recompute (ignore cache)",
                        variable=self.force_recompute_var).pack(side='left', padx=2)
        ttk.Button(btns, text="Clear Cache",
                   command=self._on_clear_cache).pack(side='left', padx=2)

        # Notebook with tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=8, pady=4)

        self.tab_overview = ttk.Frame(self.notebook)
        self.tab_check_a = ttk.Frame(self.notebook)
        self.tab_sigma = ttk.Frame(self.notebook)
        self.tab_check_b = ttk.Frame(self.notebook)
        self.tab_check_c = ttk.Frame(self.notebook)
        self.tab_viewer = ttk.Frame(self.notebook)
        self.tab_report = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_overview, text='Overview')
        self.notebook.add(self.tab_check_a, text='Check A - Scale Chain')
        self.notebook.add(self.tab_sigma, text='Sigma Trends')
        self.notebook.add(self.tab_check_b, text='Check B - Alignment')
        self.notebook.add(self.tab_check_c, text='Check C - Coverage')
        self.notebook.add(self.tab_viewer, text='Sample Viewer')
        self.notebook.add(self.tab_report, text='Report')

        # Overview tab
        self.overview_text = tk.Text(self.tab_overview, wrap='word', height=20)
        self.overview_text.pack(fill='both', expand=True, padx=6, pady=6)
        self.overview_text.insert('1.0', "Run a check to populate this tab.")
        self.overview_text.config(state='disabled')

        # Each plot tab gets a placeholder; populated after run
        self._plot_frame_a = ttk.Frame(self.tab_check_a)
        self._plot_frame_a.pack(fill='both', expand=True)
        self._plot_frame_sigma = ttk.Frame(self.tab_sigma)
        self._plot_frame_sigma.pack(fill='both', expand=True)
        self._plot_frame_b = ttk.Frame(self.tab_check_b)
        self._plot_frame_b.pack(fill='both', expand=True)
        self._plot_frame_c = ttk.Frame(self.tab_check_c)
        self._plot_frame_c.pack(fill='both', expand=True)

        self.report_text = tk.Text(self.tab_report, wrap='word', font=('Consolas', 9))
        self.report_text.pack(fill='both', expand=True, padx=6, pady=6)

        # Status + progress
        bar = ttk.Frame(self)
        bar.pack(fill='x', padx=8, pady=(0, 8))
        ttk.Label(bar, textvariable=self.status_var).pack(side='left', fill='x', expand=True)
        self.progress_bar = ttk.Progressbar(
            bar, variable=self.progress_var, maximum=100.0, length=200)
        self.progress_bar.pack(side='right')

    # ── Browse handlers ──────────────────────────────────────────────────

    def _browse_config(self):
        path = filedialog.askopenfilename(
            title="Select training_config.yaml or checkpoint .pth",
            filetypes=[("Config / Checkpoint", "*.yaml *.yml *.pth *.pt"),
                       ("All files", "*.*")])
        if path:
            self.config_var.set(path)

    def _browse_synth(self):
        path = filedialog.askdirectory(title="Select synthetic dataset folder (the <ts>_<name>/)")
        if path:
            self.synth_var.set(path)
            self._refresh_n_default()

    def _browse_calib(self):
        path = filedialog.askdirectory(title="Select calibration z-stack folder (.cine + positions.csv)")
        if path:
            self.calib_var.set(path)

    def _browse_inference(self):
        path = filedialog.askdirectory(title="Select inference crop folder (PNGs, optional defocus in filename)")
        if path:
            self.inference_var.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output folder for report")
        if path:
            self.output_var.set(path)

    def _on_open_output(self):
        path = Path(self.output_var.get())
        if not path.is_dir():
            messagebox.showinfo("Not yet", "Output folder will exist after a successful run.")
            return
        import os
        import subprocess
        if sys.platform == 'win32':
            os.startfile(str(path))
        elif sys.platform == 'darwin':
            subprocess.run(['open', str(path)])
        else:
            subprocess.run(['xdg-open', str(path)])

    def _on_clear_cache(self):
        n = clear_cache(DEFAULT_CACHE_DIR)
        messagebox.showinfo(
            "Cache cleared",
            f"Removed {n} cached fingerprint file(s) from {DEFAULT_CACHE_DIR}")

    # ── Run loop ─────────────────────────────────────────────────────────

    def _on_run(self):
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress.")
            return
        cfg = self.config_var.get().strip()
        if not cfg or not Path(cfg).is_file():
            messagebox.showerror("Missing config", "Pick a training_config.yaml or .pth checkpoint.")
            return
        synth = self.synth_var.get().strip() or None
        calib = self.calib_var.get().strip() or None
        inference = self.inference_var.get().strip() or None

        def _opt_int(s, label):
            s = (s or "").strip()
            if not s:
                return None
            try:
                return int(s)
            except ValueError:
                raise ValueError(f"{label} must be a positive integer or blank")

        try:
            n_synth = _opt_int(self.n_synth_var.get(), "Synth N")
            n_inf = _opt_int(self.n_inference_var.get(), "Inference N")
        except ValueError as e:
            messagebox.showerror("Bad N", str(e))
            return
        try:
            k = int(self.k_var.get())
            focus_offset = float(self.focus_offset_var.get())
        except ValueError:
            messagebox.showerror("Bad number", "K and focus offset must be numeric.")
            return

        self.run_button.config(state='disabled')
        self.save_button.config(state='disabled')
        self.progress_var.set(0.0)
        self.status_var.set("Starting...")
        self._run_start_time = time.monotonic()
        force_recompute = bool(self.force_recompute_var.get())

        def progress_cb(msg, frac):
            self._msg_queue.put(('progress', msg, frac * 100.0))

        def worker():
            try:
                result = run_all_checks(
                    config_path=Path(cfg),
                    synthetic_dataset_path=Path(synth) if synth else None,
                    calibration_path=Path(calib) if calib else None,
                    inference_crops_path=Path(inference) if inference else None,
                    n_synthetic_samples=n_synth,
                    n_inference_samples=n_inf,
                    k_neighbours=k,
                    calibration_focus_offset_mm=focus_offset,
                    progress=progress_cb,
                    cache_dir=DEFAULT_CACHE_DIR,
                    force_recompute=force_recompute,
                )
                self._msg_queue.put(('done', result))
            except Exception as e:
                import traceback
                self._msg_queue.put(('error', f"{e}\n{traceback.format_exc()}"))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _poll_messages(self):
        try:
            while True:
                msg = self._msg_queue.get_nowait()
                kind = msg[0]
                if kind == 'progress':
                    _, m, pct = msg
                    self.status_var.set(self._format_progress(m, pct))
                    self.progress_var.set(pct)
                elif kind == 'done':
                    self._on_run_done(msg[1])
                elif kind == 'error':
                    messagebox.showerror("Run failed", msg[1])
                    self.status_var.set("Failed.")
                    self.run_button.config(state='normal')
        except queue.Empty:
            pass
        self.after(100, self._poll_messages)

    def _format_progress(self, msg: str, pct: float) -> str:
        """Append elapsed + ETA to a progress message."""
        if self._run_start_time is None:
            return msg
        elapsed = time.monotonic() - self._run_start_time
        if pct <= 0.5 or elapsed < 1.0:
            return f"{msg}  [{int(elapsed)}s elapsed]"
        total_est = elapsed * 100.0 / pct
        remaining = max(0.0, total_est - elapsed)
        return (f"{msg}  [{int(elapsed)}s elapsed, "
                f"~{int(remaining)}s left]")

    def _on_run_done(self, result: AllChecksResult):
        self._result = result
        self.run_button.config(state='normal')
        self.save_button.config(state='normal')
        self.progress_var.set(100.0)
        self.status_var.set("Done. Inspect tabs; click Save Report to write to disk.")
        self._populate_overview(result)
        self._populate_plot_tabs(result)
        self._populate_report_tab(result)

    # ── Tab population ──────────────────────────────────────────────────

    def _populate_overview(self, result: AllChecksResult):
        lines = []
        lines.append("RUN SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Synthetic dataset:   {result.synthetic_dataset_root or '(not provided)'}")
        lines.append(f"Calibration stack:   {result.calibration_root or '(not provided)'}")
        lines.append(f"Synthetic samples:   {len(result.synthetic_fingerprints)}")
        lines.append(f"Calibration frames:  {len(result.calibration_fingerprints)}")
        lines.append("")
        lines.append("CHECK A — Scale chain")
        if result.scale_chain is None:
            lines.append("  (skipped)")
        else:
            sc = result.scale_chain
            flag = "PASS" if sc.overall_passed else "FAIL"
            lines.append(f"  Status:   {flag}")
            lines.append(f"  Tolerance: {sc.config_summary.get('tolerance_mm', '-')} mm")
            lines.append(f"  Test points: {len(sc.points)}")
            for d in sc.diagnostics:
                lines.append(f"  ! {d}")
        lines.append("")
        lines.append("CHECK B-internal — Sigma trends")
        if not result.sigma_trend_correlations:
            lines.append("  (no synthetic data)")
        else:
            for m, info in result.sigma_trend_correlations.items():
                lines.append(f"  {m:30s}  r = {info['pearson_r']:+.3f}  "
                             f"(expected {info['expected_direction']})  "
                             f"n = {info['n_finite']}")
        lines.append("")
        def _alignment_block(name, ar, flags):
            lines.append(f"CHECK B - {name}")
            if ar is None:
                lines.append("  (skipped)")
                return
            lines.append(f"  Anchors: {len(ar.comparisons)}, K={ar.k_neighbours}")
            for m, summary in ar.per_metric_summary.items():
                flag = flags.get(m, '?')
                lines.append(f"  [{flag:>4}]  {m:30s}  mean = {summary['mean_delta']:+.4f}  "
                             f"|d| = {summary['abs_mean_delta']:.4f}")

        _alignment_block(
            "Synthetic vs Calibration", result.alignment_synth_vs_calib,
            result.alignment_flags)
        lines.append("")

        def _coverage_block(name, cov, flags):
            lines.append(f"CHECK C - {name}")
            if cov is None:
                lines.append("  (skipped)")
                return
            lines.append(f"  n_test={cov.n_test}, n_reference={cov.n_reference}")
            for m, info in cov.per_feature.items():
                flag = flags.get(m, '?')
                lines.append(
                    f"  [{flag:>4}]  {m:30s}  coverage = {info['coverage_pct']:5.1f}%")

        _coverage_block(
            "Synthetic vs Inference coverage", result.coverage_synth_vs_inference,
            result.coverage_inference_flags)
        if result.joint_coverage_inference:
            jc = result.joint_coverage_inference.get('joint_coverage_pct')
            if isinstance(jc, float) and np.isfinite(jc):
                lines.append(f"  Joint multivariate coverage: {jc:.1f}%  "
                             f"(over {result.joint_coverage_inference.get('n_features_used', '?')} features)")
        lines.append("")

        if result.diagnostics:
            lines.append("DIAGNOSTICS")
            for d in result.diagnostics:
                lines.append(f"  - {d}")
        self.overview_text.config(state='normal')
        self.overview_text.delete('1.0', tk.END)
        self.overview_text.insert('1.0', "\n".join(lines))
        self.overview_text.config(state='disabled')

    def _embed_figure(self, fig, parent_frame: ttk.Frame):
        """Clear `parent_frame` and embed `fig` with a navigation toolbar."""
        for child in parent_frame.winfo_children():
            child.destroy()
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, parent_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side='bottom', fill='x')
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        self._embedded_canvases.append((canvas, parent_frame))

    def _populate_plot_tabs(self, result: AllChecksResult):
        # Check A: residuals bar
        if result.scale_chain is not None:
            self._embed_figure(plot_scale_chain_residuals(result.scale_chain),
                               self._plot_frame_a)
        # Sigma trends 4-panel
        if not result.synthetic_fingerprints.empty:
            self._embed_figure(
                plot_sigma_trends(result.synthetic_fingerprints,
                                  correlations=result.sigma_trend_correlations),
                self._plot_frame_sigma)
        # Check B: synthetic ↔ calibration alignment (per-metric deltas + heatmap)
        for child in self._plot_frame_b.winfo_children():
            child.destroy()
        ar = result.alignment_synth_vs_calib
        if ar is None:
            ttk.Label(self._plot_frame_b, text="(not run — provide a calibration folder)"
                      ).pack(pady=20)
        else:
            top_frame = ttk.Frame(self._plot_frame_b)
            top_frame.pack(fill='both', expand=True)
            bot_frame = ttk.Frame(self._plot_frame_b)
            bot_frame.pack(fill='both', expand=True)
            self._embed_figure(
                plot_alignment_per_metric_deltas(ar, result.alignment_flags),
                top_frame)
            self._embed_figure(plot_alignment_per_anchor_heatmap(ar), bot_frame)

        # Check C: synthetic ↔ inference coverage
        for child in self._plot_frame_c.winfo_children():
            child.destroy()
        cov = result.coverage_synth_vs_inference
        if cov is None:
            ttk.Label(self._plot_frame_c, text="(not run — provide an inference crops folder)"
                      ).pack(pady=20)
        else:
            self._embed_figure(
                plot_coverage_bars(cov, result.coverage_inference_flags),
                self._plot_frame_c)

        # Sample viewer — destroy any previous, build fresh
        for child in self.tab_viewer.winfo_children():
            child.destroy()
        viewer = SampleViewerFrame(self.tab_viewer, result)
        viewer.pack(fill='both', expand=True)

    def _populate_report_tab(self, result: AllChecksResult):
        # Render the markdown report inline (don't write to disk yet — Save handles that)
        from tools.fingerprint_checker.fingerprint_report import write_markdown_report
        # We need the content without writing — call the writer with a temp path
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = write_markdown_report(result, Path(td))
            text = p.read_text(encoding='utf-8')
        self.report_text.delete('1.0', tk.END)
        self.report_text.insert('1.0', text)

    # ── Save report ──────────────────────────────────────────────────────

    def _on_save(self):
        if self._result is None:
            return
        out = Path(self.output_var.get())
        try:
            files = write_full_report(self._result, out)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        msg = f"Saved {len(files)} files to:\n{out}"
        for kind, p in files.items():
            msg += f"\n  {kind}: {Path(p).name}"
        messagebox.showinfo("Report saved", msg)


class SampleViewerFrame(ttk.Frame):
    """Browse individual samples; show side-by-side closest match in another source.

    Reads from an AllChecksResult — no recompute beyond on-demand image loads
    and per-click NN lookup.
    """

    DISPLAY_PX = 256

    SOURCE_LABELS = (
        ('synthetic', 'Synthetic'),
        ('calibration', 'Calibration'),
        ('inference', 'Inference'),
    )

    METRIC_DISPLAY_ORDER = (
        ('defocus_mm', 'defocus (mm)'),
        ('sigma_px_metadata', 'meta σ_px'),
        ('erf_sigma_px', 'ERF σ (px)'),
        ('erf_r_squared', 'ERF R²'),
        ('edge_transition_width', 'edge width 10-90%'),
        ('edge_gradient_max', 'edge gradient max'),
        ('quadrant_sigma_max_min_ratio', 'quadrant σ ratio'),
        ('edge_symmetry_lr_l1', 'edge symmetry LR'),
        ('edge_symmetry_tb_l1', 'edge symmetry TB'),
        ('laplacian_variance', 'Laplacian var'),
        ('tenengrad', 'Tenengrad'),
        ('high_freq_energy_ratio', 'HF energy ratio'),
        ('background_mean', 'bg mean'),
        ('background_std', 'bg std'),
        ('object_bg_contrast', 'obj/bg contrast'),
        ('object_diameter_px', 'object diameter (px)'),
        ('centre_offset_px', 'centre offset (px)'),
        ('crop_occupancy', 'crop occupancy'),
        ('polarity', 'polarity'),
    )

    def __init__(self, parent, result: AllChecksResult):
        super().__init__(parent)
        self._result = result
        self._sources = self._build_source_map(result)
        # Cached PhotoImages (Tk garbage-collects them otherwise)
        self._photo_left = None
        self._photo_right = None
        # Cached calibration arrays (slow to load via pyphantom)
        self._image_cache: dict = {}

        self.source_var = tk.StringVar(value='synthetic')
        self.match_var = tk.StringVar(value='calibration')
        self.filter_var = tk.StringVar(value='')

        self._build_ui()
        # Populate from default source on construction
        self._on_source_changed()

    def _build_source_map(self, result):
        return {
            'synthetic': result.synthetic_fingerprints,
            'calibration': result.calibration_fingerprints,
            'inference': result.inference_fingerprints,
        }

    # ── UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top: source selector + filter
        top = ttk.Frame(self)
        top.pack(fill='x', padx=4, pady=4)
        ttk.Label(top, text="Source:").pack(side='left')
        for key, label in self.SOURCE_LABELS:
            df = self._sources.get(key)
            count = 0 if df is None else len(df)
            text = f"{label} ({count})"
            state = 'normal' if count > 0 else 'disabled'
            rb = ttk.Radiobutton(
                top, text=text, variable=self.source_var, value=key,
                command=self._on_source_changed, state=state)
            rb.pack(side='left', padx=4)
        ttk.Label(top, text="    Filter:").pack(side='left')
        ttk.Entry(top, textvariable=self.filter_var, width=24).pack(side='left')
        ttk.Button(top, text="Apply", command=self._on_source_changed).pack(
            side='left', padx=2)

        # Middle: PanedWindow horizontal split
        pw = ttk.PanedWindow(self, orient='horizontal')
        pw.pack(fill='both', expand=True, padx=4, pady=4)

        # ── Left pane: sample list + selected sample image/fingerprint ──
        left = ttk.Frame(pw)
        pw.add(left, weight=1)

        list_frame = ttk.LabelFrame(left, text="Samples", padding=4)
        list_frame.pack(fill='both', expand=True)
        self.tree = ttk.Treeview(
            list_frame,
            columns=('idx', 'defocus', 'sigma_meta', 'erf_sigma'),
            show='headings', height=10,
        )
        self.tree.heading('idx', text='#')
        self.tree.heading('defocus', text='defocus (mm)')
        self.tree.heading('sigma_meta', text='meta σ')
        self.tree.heading('erf_sigma', text='ERF σ')
        self.tree.column('idx', width=40, anchor='e')
        self.tree.column('defocus', width=90, anchor='e')
        self.tree.column('sigma_meta', width=70, anchor='e')
        self.tree.column('erf_sigma', width=70, anchor='e')
        sb = ttk.Scrollbar(list_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self.tree.bind('<<TreeviewSelect>>', self._on_select)

        sel_image_frame = ttk.LabelFrame(left, text="Selected sample", padding=4)
        sel_image_frame.pack(fill='x', pady=(4, 0))
        self.left_canvas = tk.Canvas(
            sel_image_frame, width=self.DISPLAY_PX, height=self.DISPLAY_PX,
            background='#1a1a1a')
        self.left_canvas.pack()
        self.left_caption_var = tk.StringVar(value='(select a sample)')
        ttk.Label(sel_image_frame, textvariable=self.left_caption_var,
                  font=('TkDefaultFont', 8), wraplength=self.DISPLAY_PX).pack(
            anchor='w', pady=(2, 0))

        # ── Right pane: match controls + matched image + delta table ──
        right = ttk.Frame(pw)
        pw.add(right, weight=2)

        match_top = ttk.Frame(right)
        match_top.pack(fill='x', pady=(0, 4))
        ttk.Label(match_top, text="Match against:").pack(side='left')
        self.match_radios = []
        for key, label in self.SOURCE_LABELS:
            df = self._sources.get(key)
            count = 0 if df is None else len(df)
            text = f"{label} ({count})"
            state = 'normal' if count > 0 else 'disabled'
            rb = ttk.Radiobutton(
                match_top, text=text, variable=self.match_var, value=key,
                command=self._on_select, state=state)
            rb.pack(side='left', padx=4)
            self.match_radios.append(rb)

        match_image_frame = ttk.LabelFrame(right, text="Closest match", padding=4)
        match_image_frame.pack(fill='x')
        self.right_canvas = tk.Canvas(
            match_image_frame, width=self.DISPLAY_PX, height=self.DISPLAY_PX,
            background='#1a1a1a')
        self.right_canvas.pack()
        self.right_caption_var = tk.StringVar(value='(no match)')
        ttk.Label(match_image_frame, textvariable=self.right_caption_var,
                  font=('TkDefaultFont', 8), wraplength=self.DISPLAY_PX).pack(
            anchor='w', pady=(2, 0))

        # Per-feature delta table
        delta_frame = ttk.LabelFrame(right, text="Per-feature comparison", padding=4)
        delta_frame.pack(fill='both', expand=True, pady=(4, 0))
        self.delta_tree = ttk.Treeview(
            delta_frame,
            columns=('metric', 'selected', 'match', 'delta'),
            show='headings', height=12,
        )
        self.delta_tree.heading('metric', text='metric')
        self.delta_tree.heading('selected', text='selected')
        self.delta_tree.heading('match', text='match')
        self.delta_tree.heading('delta', text='Δ (match − selected)')
        self.delta_tree.column('metric', width=180)
        self.delta_tree.column('selected', width=110, anchor='e')
        self.delta_tree.column('match', width=110, anchor='e')
        self.delta_tree.column('delta', width=160, anchor='e')
        sb2 = ttk.Scrollbar(delta_frame, orient='vertical',
                            command=self.delta_tree.yview)
        self.delta_tree.configure(yscrollcommand=sb2.set)
        self.delta_tree.pack(side='left', fill='both', expand=True)
        sb2.pack(side='right', fill='y')

    # ── Behaviour ────────────────────────────────────────────────────────

    def _current_source_df(self):
        return self._sources.get(self.source_var.get())

    def _current_match_df(self):
        return self._sources.get(self.match_var.get())

    def _filter_df(self, df):
        text = self.filter_var.get().strip().lower()
        if not text or df is None or df.empty:
            return df
        # Search across source_path + key numeric columns coerced to string
        cols = [c for c in ('source_path', 'defocus_mm', 'sigma_px_metadata',
                            'erf_sigma_px', 'camera') if c in df.columns]
        mask = pd.Series(False, index=df.index)
        for c in cols:
            mask |= df[c].astype(str).str.lower().str.contains(text, na=False)
        return df[mask]

    def _on_source_changed(self):
        df = self._current_source_df()
        df = self._filter_df(df)
        # Repopulate the tree
        self.tree.delete(*self.tree.get_children())
        if df is None or df.empty:
            return
        for idx, row in df.iterrows():
            d = row.get('defocus_mm', float('nan'))
            sm = row.get('sigma_px_metadata', float('nan'))
            erf = row.get('erf_sigma_px', float('nan'))
            self.tree.insert('', 'end', iid=str(idx), values=(
                int(idx),
                f"{d:+.3f}" if pd.notna(d) else '-',
                f"{sm:.3f}" if pd.notna(sm) else '-',
                f"{erf:.3f}" if pd.notna(erf) else '-',
            ))
        # Auto-pick first row to populate panels
        first = self.tree.get_children()
        if first:
            self.tree.selection_set(first[0])
            self.tree.see(first[0])

    def _on_select(self, _event=None):
        df = self._current_source_df()
        if df is None or df.empty:
            return
        sel = self.tree.selection()
        if not sel:
            return
        try:
            idx = int(sel[0])
        except ValueError:
            return
        if idx not in df.index:
            return
        row = df.loc[idx]
        self._render_left(row)
        self._render_match_and_delta(row)

    def _render_left(self, row):
        img = self._load_image_cached(row)
        self._photo_left = self._array_to_photoimage(img)
        self._draw_canvas(self.left_canvas, self._photo_left)
        self.left_caption_var.set(self._format_caption(row))

    def _render_match_and_delta(self, source_row):
        target_df = self._current_match_df()
        # Don't match against the same source
        if (target_df is None or target_df.empty
                or self.match_var.get() == self.source_var.get()):
            self._photo_right = None
            self._draw_canvas(self.right_canvas, None)
            self.right_caption_var.set("(pick a different match source)")
            self.delta_tree.delete(*self.delta_tree.get_children())
            return
        target_idx, distance = find_nearest_match(source_row, target_df)
        if target_idx is None:
            self._photo_right = None
            self._draw_canvas(self.right_canvas, None)
            self.right_caption_var.set("(no comparable features)")
            self.delta_tree.delete(*self.delta_tree.get_children())
            return
        target_row = target_df.loc[target_idx]
        img = self._load_image_cached(target_row)
        self._photo_right = self._array_to_photoimage(img)
        self._draw_canvas(self.right_canvas, self._photo_right)
        cap = self._format_caption(target_row)
        cap += f"\n  z-scored distance: {distance:.3f}"
        self.right_caption_var.set(cap)
        # Populate delta table
        self.delta_tree.delete(*self.delta_tree.get_children())
        for col, label in self.METRIC_DISPLAY_ORDER:
            sel_v = source_row.get(col)
            mat_v = target_row.get(col)
            sel_s, mat_s, delta_s = self._format_metric_row(sel_v, mat_v)
            self.delta_tree.insert(
                '', 'end',
                values=(label, sel_s, mat_s, delta_s),
            )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _load_image_cached(self, row):
        path = row.get('source_path', '')
        if not path:
            return None
        if path in self._image_cache:
            return self._image_cache[path]
        img = load_sample_image_by_row(row)
        # Cache only calibration (slow to reload); PNG reads are fast enough
        if path.lower().endswith('.cine') and img is not None:
            # Cap cache size
            if len(self._image_cache) >= 30:
                self._image_cache.pop(next(iter(self._image_cache)))
            self._image_cache[path] = img
        return img

    def _array_to_photoimage(self, arr):
        if arr is None:
            return None
        try:
            from PIL import Image, ImageTk
        except ImportError:
            return None
        if arr.ndim != 2:
            return None
        a = arr
        if a.dtype != np.uint8:
            a = np.clip(a * 255.0, 0, 255).astype(np.uint8) \
                if a.max() <= 1.5 else a.astype(np.uint8)
        pil = Image.fromarray(a, mode='L')
        pil = pil.resize((self.DISPLAY_PX, self.DISPLAY_PX),
                         resample=Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil)

    def _draw_canvas(self, canvas, photo):
        canvas.delete('all')
        if photo is None:
            canvas.create_text(
                self.DISPLAY_PX // 2, self.DISPLAY_PX // 2,
                text='(no image)', fill='#888888')
        else:
            canvas.create_image(0, 0, anchor='nw', image=photo)

    @staticmethod
    def _format_caption(row):
        parts = []
        st = row.get('source_type', '')
        idx = row.get('index', '?')
        parts.append(f"[{st}] #{idx}")
        d = row.get('defocus_mm')
        if pd.notna(d):
            parts.append(f"z = {d:+.3f} mm")
        sm = row.get('sigma_px_metadata')
        if pd.notna(sm):
            parts.append(f"meta σ = {sm:.3f}")
        path = row.get('source_path', '')
        if path:
            parts.append(Path(path).name)
        return "  ·  ".join(parts)

    @staticmethod
    def _format_metric_row(sel, mat):
        def _fmt(v):
            if isinstance(v, str):
                return v
            try:
                if v is None or pd.isna(v):
                    return '-'
                return f"{float(v):.4f}"
            except (TypeError, ValueError):
                return str(v)
        if isinstance(sel, str) or isinstance(mat, str):
            # Categorical (e.g. polarity)
            delta = '(match)' if sel == mat else '(differ)'
            return _fmt(sel), _fmt(mat), delta
        sel_s, mat_s = _fmt(sel), _fmt(mat)
        try:
            delta_v = float(mat) - float(sel)
            if not np.isfinite(delta_v):
                return sel_s, mat_s, '-'
            ref = abs(float(sel)) if pd.notna(sel) and abs(float(sel)) > 1e-9 else None
            if ref is not None:
                pct = 100.0 * delta_v / ref
                return sel_s, mat_s, f"{delta_v:+.4f}  ({pct:+.1f}%)"
            return sel_s, mat_s, f"{delta_v:+.4f}"
        except (TypeError, ValueError):
            return sel_s, mat_s, '-'


def main():
    app = FingerprintCheckerApp()
    app.mainloop()


if __name__ == '__main__':
    main()
