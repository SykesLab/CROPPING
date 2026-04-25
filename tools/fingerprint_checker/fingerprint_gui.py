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
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

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

from tools.fingerprint_checker.fingerprint_orchestrator import (  # noqa: E402
    AllChecksResult, run_all_checks,
)
from tools.fingerprint_checker.fingerprint_plots import (  # noqa: E402
    plot_alignment_per_anchor_heatmap,
    plot_alignment_per_metric_deltas,
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
        self.output_var = tk.StringVar(value=str(_REPO_ROOT / "tools" / "fingerprint_checker" / "output" / datetime.now().strftime("%Y%m%d_%H%M%S")))
        self.n_synth_var = tk.StringVar(value=str(self._dataset_count(self.synth_var.get())))
        self.k_var = tk.StringVar(value="20")
        self.focus_offset_var = tk.StringVar(value="0.0")

        self.status_var = tk.StringVar(value="Ready. Pick a config + synthetic dataset (and optionally calibration), then Run All.")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._result: Optional[AllChecksResult] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._msg_queue: "queue.Queue" = queue.Queue()
        self._embedded_canvases = []  # (FigureCanvasTkAgg, frame) list for cleanup

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

    def _refresh_n_default(self, *_):
        self.n_synth_var.set(str(self._dataset_count(self.synth_var.get())))

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
        _row("Output folder:", self.output_var, self._browse_output)

        opts = ttk.Frame(top)
        opts.pack(fill='x', pady=(4, 2))
        ttk.Label(opts, text="Synthetic samples to fingerprint:").pack(side='left', padx=(0, 4))
        ttk.Entry(opts, textvariable=self.n_synth_var, width=10).pack(side='left')
        ttk.Label(opts, text=" (full dataset shown by default; reduce for fast iteration)",
                  foreground='gray', font=('TkDefaultFont', 8)).pack(side='left', padx=4)

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

        # Notebook with tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=8, pady=4)

        self.tab_overview = ttk.Frame(self.notebook)
        self.tab_check_a = ttk.Frame(self.notebook)
        self.tab_sigma = ttk.Frame(self.notebook)
        self.tab_check_b = ttk.Frame(self.notebook)
        self.tab_report = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_overview, text='Overview')
        self.notebook.add(self.tab_check_a, text='Check A — Scale Chain')
        self.notebook.add(self.tab_sigma, text='Sigma Trends')
        self.notebook.add(self.tab_check_b, text='Check B — Alignment')
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
        try:
            n = int(self.n_synth_var.get()) if self.n_synth_var.get() else None
        except ValueError:
            messagebox.showerror("Bad N", "Synthetic sample count must be an integer.")
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

        def progress_cb(msg, frac):
            self._msg_queue.put(('progress', msg, frac * 100.0))

        def worker():
            try:
                result = run_all_checks(
                    config_path=Path(cfg),
                    synthetic_dataset_path=Path(synth) if synth else None,
                    calibration_path=Path(calib) if calib else None,
                    n_synthetic_samples=n,
                    k_neighbours=k,
                    calibration_focus_offset_mm=focus_offset,
                    progress=progress_cb,
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
                    self.status_var.set(m)
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
        lines.append("CHECK B — Synthetic vs Calibration alignment")
        if result.alignment_synth_vs_calib is None:
            lines.append("  (skipped — no calibration provided or empty fingerprints)")
        else:
            ar = result.alignment_synth_vs_calib
            lines.append(f"  Anchors (calibration): {len(ar.comparisons)}")
            lines.append(f"  K neighbours per anchor: {ar.k_neighbours}")
            for m, summary in ar.per_metric_summary.items():
                flag = result.alignment_flags.get(m, '?')
                lines.append(f"  [{flag:>4}]  {m:30s}  mean Δ = {summary['mean_delta']:+.4f}  "
                             f"|Δ| = {summary['abs_mean_delta']:.4f}")
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
        # Check B: per-metric bars + heatmap (stack vertically in same tab)
        if result.alignment_synth_vs_calib is not None:
            for child in self._plot_frame_b.winfo_children():
                child.destroy()
            top_frame = ttk.Frame(self._plot_frame_b)
            top_frame.pack(fill='both', expand=True)
            bot_frame = ttk.Frame(self._plot_frame_b)
            bot_frame.pack(fill='both', expand=True)
            self._embed_figure(
                plot_alignment_per_metric_deltas(
                    result.alignment_synth_vs_calib, result.alignment_flags),
                top_frame)
            self._embed_figure(
                plot_alignment_per_anchor_heatmap(result.alignment_synth_vs_calib),
                bot_frame)

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


def main():
    app = FingerprintCheckerApp()
    app.mainloop()


if __name__ == '__main__':
    main()
