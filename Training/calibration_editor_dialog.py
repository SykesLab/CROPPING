"""Modal dialog: edit (ρ, σ₀) on a checkpoint and save a new edited version.

Opened from Tab 5 in the training GUI. Can be launched blank (user picks a
checkpoint) or with prefilled values (e.g. from a post-hoc fit on Tab 5).

Safety: refuses to overwrite checkpoints under runs/. Always writes new
edits to training_output/real_crop_validation/<run_name>/edits/.
"""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

from calibration_editor import (
    CalibrationError, CalibrationSnapshot,
    apply_linear_correction,
    load_checkpoint, next_edit_filename, read_calibration, read_history,
    save_corrected_checkpoint,
)


class CalibrationEditorDialog(tk.Toplevel):
    """Toplevel dialog to inspect and edit checkpoint calibration constants."""

    def __init__(self, parent, training_output_root: Path,
                 initial_checkpoint: Optional[Path] = None,
                 prefill_a: Optional[float] = None,
                 prefill_b: Optional[float] = None,
                 on_saved: Optional[Callable[[Path], None]] = None):
        super().__init__(parent)
        self.title("Calibration Editor")
        self.transient(parent)
        self.grab_set()
        self.resizable(False, True)

        self.training_output_root = Path(training_output_root)
        self.on_saved = on_saved

        # Loaded-checkpoint state
        self._checkpoint_path: Optional[Path] = None
        self._history: list = []
        self._rho_loaded: Optional[float] = None
        self._sigma_0_loaded: Optional[float] = None
        self._run_name: Optional[str] = None
        self._mode: Optional[str] = None

        # Tk vars
        self.ckpt_var = tk.StringVar()
        self.rho_var = tk.StringVar()
        self.sigma_0_var = tk.StringVar()
        self.note_var = tk.StringVar()
        self.info_var = tk.StringVar(value="(no checkpoint loaded)")
        self.preview_var = tk.StringVar(value="")
        self.a_var = tk.StringVar(value="1.0")
        self.b_var = tk.StringVar(value="0.0")

        self._build_ui()

        if prefill_a is not None:
            self.a_var.set(f"{prefill_a:.6g}")
        if prefill_b is not None:
            self.b_var.set(f"{prefill_b:.6g}")

        if initial_checkpoint is not None:
            self.ckpt_var.set(str(initial_checkpoint))
            self._load_checkpoint_from_var()

    # ── UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {'padx': 10, 'pady': 4}

        row = ttk.Frame(self)
        row.pack(fill='x', **pad)
        ttk.Label(row, text="Checkpoint:", width=12).pack(side='left')
        ttk.Entry(row, textvariable=self.ckpt_var, width=60).pack(
            side='left', padx=(0, 5))
        ttk.Button(row, text="Browse...", command=self._browse).pack(side='left')
        ttk.Button(row, text="Load", command=self._load_checkpoint_from_var).pack(
            side='left', padx=4)

        ttk.Label(self, textvariable=self.info_var,
                  foreground='gray', font=('TkDefaultFont', 8)).pack(
            anchor='w', padx=12)

        vals_frame = ttk.LabelFrame(self, text="Calibration constants", padding=8)
        vals_frame.pack(fill='x', **pad)
        r1 = ttk.Frame(vals_frame)
        r1.pack(fill='x', pady=2)
        ttk.Label(r1, text="ρ (px/mm):", width=14).pack(side='left')
        ttk.Entry(r1, textvariable=self.rho_var, width=16).pack(side='left')
        r2 = ttk.Frame(vals_frame)
        r2.pack(fill='x', pady=2)
        ttk.Label(r2, text="σ₀ (px):", width=14).pack(side='left')
        ttk.Entry(r2, textvariable=self.sigma_0_var, width=16).pack(side='left')

        ttk.Label(vals_frame, text="Preview (σ → z):",
                  font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', pady=(6, 0))
        ttk.Label(vals_frame, textvariable=self.preview_var,
                  font=('Consolas', 9), justify='left').pack(anchor='w')

        self.rho_var.trace_add('write', lambda *a: self._update_preview())
        self.sigma_0_var.trace_add('write', lambda *a: self._update_preview())

        fit_frame = ttk.LabelFrame(
            self, text="Auto-fill from linear fit  (ẑ_corr = a·ẑ + b)", padding=8)
        fit_frame.pack(fill='x', **pad)
        r3 = ttk.Frame(fit_frame)
        r3.pack(fill='x', pady=2)
        ttk.Label(r3, text="slope a:", width=10).pack(side='left')
        ttk.Entry(r3, textvariable=self.a_var, width=10).pack(
            side='left', padx=(0, 8))
        ttk.Label(r3, text="offset b:", width=10).pack(side='left')
        ttk.Entry(r3, textvariable=self.b_var, width=10).pack(side='left')
        ttk.Button(fit_frame, text="Apply fit → ρ, σ₀",
                   command=self._apply_fit).pack(anchor='w', pady=(4, 0))

        hist_frame = ttk.LabelFrame(self, text="History", padding=8)
        hist_frame.pack(fill='both', expand=True, **pad)
        columns = ('ts', 'source', 'rho', 'sigma_0', 'note')
        self.history_tree = ttk.Treeview(
            hist_frame, columns=columns, show='headings', height=5)
        self.history_tree.heading('ts', text='Timestamp')
        self.history_tree.heading('source', text='Source')
        self.history_tree.heading('rho', text='ρ (px/mm)')
        self.history_tree.heading('sigma_0', text='σ₀ (px)')
        self.history_tree.heading('note', text='Note')
        self.history_tree.column('ts', width=140)
        self.history_tree.column('source', width=70)
        self.history_tree.column('rho', width=90, anchor='e')
        self.history_tree.column('sigma_0', width=90, anchor='e')
        self.history_tree.column('note', width=300)
        self.history_tree.pack(fill='both', expand=True, side='left')
        sb = ttk.Scrollbar(
            hist_frame, orient='vertical', command=self.history_tree.yview)
        sb.pack(side='right', fill='y')
        self.history_tree.configure(yscrollcommand=sb.set)
        ttk.Button(self, text="Revert ρ, σ₀ to selected history entry",
                   command=self._revert_to_selected).pack(anchor='w', **pad)

        row = ttk.Frame(self)
        row.pack(fill='x', **pad)
        ttk.Label(row, text="Note:", width=12).pack(side='left')
        ttk.Entry(row, textvariable=self.note_var, width=70).pack(side='left')

        btns = ttk.Frame(self)
        btns.pack(fill='x', **pad)
        ttk.Button(btns, text="Save as new...",
                   command=self._save_as_new).pack(side='left', padx=2)
        ttk.Button(btns, text="Save (overwrite)",
                   command=self._save_overwrite).pack(side='left', padx=2)
        ttk.Button(btns, text="Close",
                   command=self.destroy).pack(side='right')

    # ── Load ─────────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pth *.pt"),
                       ("All files", "*.*")],
            parent=self,
        )
        if path:
            self.ckpt_var.set(path)
            self._load_checkpoint_from_var()

    def _load_checkpoint_from_var(self):
        raw = self.ckpt_var.get().strip()
        if not raw:
            return
        path = Path(raw)
        if not path.is_file():
            messagebox.showerror("Not found",
                                 f"Checkpoint file does not exist:\n{path}",
                                 parent=self)
            return
        try:
            ckpt = load_checkpoint(path)
            rho, sigma_0 = read_calibration(ckpt)
        except CalibrationError as e:
            messagebox.showerror("Invalid checkpoint", str(e), parent=self)
            return
        except Exception as e:
            messagebox.showerror("Load failed",
                                 f"Could not load checkpoint:\n{e}", parent=self)
            return

        self._checkpoint_path = path
        self._rho_loaded = rho
        self._sigma_0_loaded = sigma_0
        self._mode = (ckpt.get('config', {}).get('training', {})
                      .get('training_mode', 'unknown'))
        self._history = read_history(ckpt)
        if not self._history:
            self._history.append(CalibrationSnapshot(
                rho=rho, sigma_0=sigma_0, source='training',
                timestamp='(from config)',
                note='Current checkpoint values — no edit history yet'))

        from run_paths import detect_run_name
        self._run_name = detect_run_name(path)
        self.rho_var.set(f"{rho:.6g}")
        self.sigma_0_var.set(f"{sigma_0:.6g}")

        size_kb = path.stat().st_size / 1024
        epoch = ckpt.get('epoch', '?')
        self.info_var.set(
            f"Mode: {self._mode}  |  epoch: {epoch}  |  size: {size_kb:.0f} KB"
            f"  |  run: {self._run_name or '(cannot infer)'}")

        if self._mode != 'direct':
            messagebox.showwarning(
                "Non-direct mode",
                f"Training mode is '{self._mode}'. ρ/σ₀ editing is only "
                "meaningful for direct-mode checkpoints. You can still edit, "
                "but inference in non-direct mode won't use these values.",
                parent=self)

        self._refresh_history_display()
        self._update_preview()

    def _refresh_history_display(self):
        self.history_tree.delete(*self.history_tree.get_children())
        for s in self._history:
            self.history_tree.insert(
                '', 'end',
                values=(s.timestamp, s.source,
                        f"{s.rho:.6g}", f"{s.sigma_0:.6g}", s.note))

    # ── Preview ──────────────────────────────────────────────────────────

    def _update_preview(self):
        try:
            rho = float(self.rho_var.get())
            sigma_0 = float(self.sigma_0_var.get())
        except (ValueError, TypeError):
            self.preview_var.set("(enter valid numbers)")
            return
        if abs(rho) < 1e-9:
            self.preview_var.set("(ρ too small to compute z)")
            return
        lines = []
        for s in (1.0, 3.0, 5.0, 7.0, 9.0):
            z_new = (s - sigma_0) / rho
            if (self._rho_loaded is not None
                    and self._sigma_0_loaded is not None
                    and abs(self._rho_loaded) > 1e-9):
                z_old = (s - self._sigma_0_loaded) / self._rho_loaded
                delta = z_new - z_old
                lines.append(
                    f"  σ = {s:4.1f} px  →  z = {z_new:7.3f} mm   "
                    f"(was {z_old:7.3f} mm,  Δ {delta:+.3f})")
            else:
                lines.append(f"  σ = {s:4.1f} px  →  z = {z_new:7.3f} mm")
        self.preview_var.set("\n".join(lines))

    # ── Actions ──────────────────────────────────────────────────────────

    def _apply_fit(self):
        try:
            a = float(self.a_var.get())
            b = float(self.b_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid fit", "slope/offset must be numeric", parent=self)
            return
        if self._rho_loaded is None:
            messagebox.showerror("No checkpoint",
                                 "Load a checkpoint first.", parent=self)
            return
        try:
            rho_new, sigma_0_new = apply_linear_correction(
                self._rho_loaded, self._sigma_0_loaded, a, b)
        except CalibrationError as e:
            messagebox.showerror("Cannot apply fit", str(e), parent=self)
            return
        self.rho_var.set(f"{rho_new:.6g}")
        self.sigma_0_var.set(f"{sigma_0_new:.6g}")
        if not self.note_var.get().strip():
            self.note_var.set(f"Linear fit a={a:.4g}, b={b:.4g}")

    def _revert_to_selected(self):
        sel = self.history_tree.selection()
        if not sel:
            messagebox.showwarning("No selection",
                                   "Select a history row first.", parent=self)
            return
        idx = self.history_tree.index(sel[0])
        if idx < 0 or idx >= len(self._history):
            return
        snap = self._history[idx]
        self.rho_var.set(f"{snap.rho:.6g}")
        self.sigma_0_var.set(f"{snap.sigma_0:.6g}")
        if not self.note_var.get().strip():
            self.note_var.set(f"Revert to {snap.timestamp} ({snap.source})")

    def _collect_edit(self):
        try:
            rho = float(self.rho_var.get())
            sigma_0 = float(self.sigma_0_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid values", "ρ and σ₀ must be numeric.", parent=self)
            return None
        if rho <= 0:
            messagebox.showerror("Invalid ρ",
                                 "ρ must be positive.", parent=self)
            return None
        if sigma_0 < 0:
            messagebox.showerror("Invalid σ₀",
                                 "σ₀ must be non-negative.", parent=self)
            return None
        return rho, sigma_0, self.note_var.get().strip() or "Manual edit"

    def _determine_source_label(self, rho, sigma_0) -> str:
        """Return 'fit' if the edit matches the current a/b entries, else 'manual'."""
        if self._rho_loaded is None:
            return 'manual'
        try:
            a_str = self.a_var.get().strip()
            b_str = self.b_var.get().strip()
            if a_str and b_str:
                a = float(a_str)
                b = float(b_str)
                if a == 1.0 and b == 0.0:
                    return 'manual'
                exp_rho, exp_sigma_0 = apply_linear_correction(
                    self._rho_loaded, self._sigma_0_loaded, a, b)
                if (abs(exp_rho - rho) < 1e-6
                        and abs(exp_sigma_0 - sigma_0) < 1e-6):
                    return 'fit'
        except Exception:
            pass
        return 'manual'

    def _save_as_new(self):
        if self._checkpoint_path is None:
            messagebox.showerror("No checkpoint",
                                 "Load a checkpoint first.", parent=self)
            return
        edit = self._collect_edit()
        if edit is None:
            return
        rho, sigma_0, note = edit
        if self._run_name is None:
            messagebox.showerror(
                "Cannot infer run",
                "Could not infer the run name from the checkpoint path "
                f"({self._checkpoint_path}).\nExpected a path under "
                "runs/<run_name>/checkpoints/ or "
                "real_crop_validation/<run_name>/edits/.",
                parent=self)
            return
        from run_paths import edits_dir
        target_dir = edits_dir(self.training_output_root, self._run_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = next_edit_filename(target_dir)

        try:
            snap = save_corrected_checkpoint(
                self._checkpoint_path, out_path, rho, sigma_0,
                source_label=self._determine_source_label(rho, sigma_0),
                note=note,
            )
        except CalibrationError as e:
            messagebox.showerror("Save failed", str(e), parent=self)
            return
        except Exception as e:
            messagebox.showerror(
                "Save failed", f"Unexpected error:\n{e}", parent=self)
            return

        messagebox.showinfo(
            "Saved",
            f"Saved corrected checkpoint to:\n{out_path}\n\n"
            f"ρ = {snap.rho:.6g},  σ₀ = {snap.sigma_0:.6g}",
            parent=self)
        if self.on_saved:
            self.on_saved(out_path)
        self.ckpt_var.set(str(out_path))
        self._load_checkpoint_from_var()

    def _save_overwrite(self):
        if self._checkpoint_path is None:
            messagebox.showerror("No checkpoint",
                                 "Load a checkpoint first.", parent=self)
            return
        parts = self._checkpoint_path.resolve().parts
        if 'runs' in parts:
            messagebox.showerror(
                "Refuse to overwrite source",
                "This checkpoint lives under runs/, which holds your trained "
                "models. Use 'Save as new...' instead — the trained checkpoint "
                "must not be mutated.", parent=self)
            return
        if not messagebox.askyesno(
            "Confirm overwrite",
            f"Replace the contents of {self._checkpoint_path.name} in place?\n\n"
            "Anything pointing at this file by path will silently start using "
            "the new ρ/σ₀. The calibration_history inside the file still "
            "preserves the audit trail.", parent=self,
        ):
            return
        edit = self._collect_edit()
        if edit is None:
            return
        rho, sigma_0, note = edit

        import shutil
        temp = self._checkpoint_path.with_suffix('.pth.tmp')
        shutil.copy2(self._checkpoint_path, temp)
        try:
            snap = save_corrected_checkpoint(
                temp, self._checkpoint_path, rho, sigma_0,
                source_label=self._determine_source_label(rho, sigma_0),
                note=note,
            )
        finally:
            if temp.exists():
                temp.unlink()

        messagebox.showinfo(
            "Saved",
            f"Overwrote {self._checkpoint_path.name}.\n\n"
            f"ρ = {snap.rho:.6g},  σ₀ = {snap.sigma_0:.6g}",
            parent=self)
        if self.on_saved:
            self.on_saved(self._checkpoint_path)
        self._load_checkpoint_from_var()
