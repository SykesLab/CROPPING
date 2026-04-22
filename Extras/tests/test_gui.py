"""
CROPPING Test Runner GUI

Standalone tkinter application for running pipeline tests interactively.
Select individual tests or run all, see pass/fail results in real-time.

Usage:
    python test_gui.py
"""

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Test registry — each test maps to a class in test_pipeline.py
# ---------------------------------------------------------------------------

TESTS: List[Dict[str, str]] = [
    {
        "id": "calibration_roundtrip",
        "name": "Calibration Round-Trip",
        "class": "TestCalibrationRoundTrip",
        "description": (
            "Verifies that the calibration forward-inverse chain is "
            "lossless: sigma_to_depth(rho * |z| + sigma_0) == |z|"
        ),
    },
    {
        "id": "dme_loss_properties",
        "name": "DME Loss Properties",
        "class": "TestDMELossProperties",
        "description": (
            "Checks that the DME loss function returns zero for identical "
            "inputs and is symmetric: L(a,b) == L(b,a)"
        ),
    },
    {
        "id": "dme_loss_relative",
        "name": "DME Loss Relative Error",
        "class": "TestDMELossRelativeError",
        "description": (
            "Validates that log-space loss is roughly scale-invariant: "
            "equal relative errors produce similar losses at low and high blur"
        ),
    },
    {
        "id": "model_output",
        "name": "Model Output Contract",
        "class": "TestModelOutputContract",
        "description": (
            "Checks that DefocusNet outputs shape (B, 1) with all values "
            "in [0, 1] (sigmoid bounded)"
        ),
    },
    {
        "id": "gaussian_kernel",
        "name": "Gaussian Kernel",
        "class": "TestGaussianKernel",
        "description": (
            "Verifies Gaussian blur kernels sum to 1.0 (normalised) "
            "and are symmetric for multiple sigma values"
        ),
    },
    {
        "id": "blur_measure_roundtrip",
        "name": "Blur Measure Round-Trip",
        "class": "TestBlurMeasureRoundTrip",
        "description": (
            "Applies a known Gaussian blur to a sharp edge, then recovers "
            "sigma via ERF fit. Checks recovery within 5% tolerance"
        ),
    },
    {
        "id": "inference_scaling",
        "name": "Inference Scaling Chain",
        "class": "TestInferenceScalingChain",
        "description": (
            "Verifies the full forward-inverse scaling chain: if the model "
            "predicts perfectly, the recovered defocus must match exactly. "
            "Tests both same-camera and cross-camera scenarios"
        ),
    },
    {
        "id": "label_normalisation",
        "name": "Label Normalisation Clamp",
        "class": "TestLabelNormalisationClamp",
        "description": (
            "Checks that normalised training labels are always in [0, 1], "
            "even when native blur exceeds max_sigma"
        ),
    },
    {
        "id": "physics_module",
        "name": "Physics Module Round-Trip",
        "class": "TestPhysicsModuleRoundTrip",
        "description": (
            "Validates the unified physics module: forward-inverse round-trip "
            "recovers exact defocus for same-camera and cross-camera scenarios. "
            "Also checks clamping/saturation flags and consistency with original inline math"
        ),
    },
]

TEST_FILE = Path(__file__).resolve().parent / "test_pipeline.py"


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_single_test(test_class: str) -> Dict[str, str]:
    """Run a single test class via pytest subprocess. Returns result dict."""
    cmd = [
        sys.executable, "-m", "pytest",
        str(TEST_FILE),
        "-v", "--timeout=10", "--tb=short", "--no-header",
        "-k", test_class,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            cwd=str(TEST_FILE.parent),
        )
        passed = result.returncode == 0
        # Extract relevant output lines
        output = result.stdout + result.stderr
        return {
            "passed": passed,
            "output": output.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": "TIMEOUT (>30s)", "returncode": -1}
    except Exception as e:
        return {"passed": False, "output": str(e), "returncode": -1}


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class TestRunnerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CROPPING Test Runner")
        self.minsize(900, 650)
        self.resizable(True, True)

        self._results: Dict[str, Optional[Dict]] = {t["id"]: None for t in TESTS}
        self._running = False

        self._build_ui()

    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 4))

        ttk.Button(top, text="Run All Tests", command=self._run_all).pack(side="left", padx=4)
        self.btn_run_selected = ttk.Button(top, text="Run Selected", command=self._run_selected)
        self.btn_run_selected.pack(side="left", padx=4)

        self.var_status = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.var_status, font=("Consolas", 9)).pack(
            side="right", padx=4
        )

        # Progress bar
        self.progress = ttk.Progressbar(top, length=200, mode="determinate")
        self.progress.pack(side="right", padx=(4, 8))

        # Main paned view: test list (left) + detail (right)
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=4)

        # Left: test list
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        columns = ("status", "name")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", selectmode="extended")
        self.tree.heading("status", text="Status")
        self.tree.heading("name", text="Test")
        self.tree.column("status", width=80, anchor="center")
        self.tree.column("name", width=250)
        self.tree.pack(fill="both", expand=True)

        for test in TESTS:
            self.tree.insert("", "end", iid=test["id"], values=("---", test["name"]))

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Right: detail panel
        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        # Description
        self.var_test_name = tk.StringVar(value="Select a test")
        ttk.Label(right, textvariable=self.var_test_name,
                  font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=8, pady=(8, 2))

        self.var_description = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.var_description,
                  wraplength=450, justify="left",
                  font=("Segoe UI", 9)).pack(anchor="w", padx=8, pady=(0, 8))

        # Result output
        ttk.Label(right, text="Output:", font=("Segoe UI", 9, "bold")).pack(
            anchor="w", padx=8
        )
        self.output_text = tk.Text(
            right, height=20, wrap="word",
            font=("Consolas", 9), state="disabled",
            bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="#d4d4d4",
        )
        self.output_text.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        # Configure text tags for coloured output
        self.output_text.tag_configure("pass", foreground="#4ec94e")
        self.output_text.tag_configure("fail", foreground="#f44747")
        self.output_text.tag_configure("info", foreground="#569cd6")

        # Run button for individual test
        btn_frame = ttk.Frame(right)
        btn_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.btn_run_one = ttk.Button(btn_frame, text="Run This Test",
                                       command=self._run_current, state="disabled")
        self.btn_run_one.pack(side="left")

        # Summary bar at bottom
        summary = ttk.Frame(self)
        summary.pack(fill="x", padx=10, pady=(0, 8))
        self.var_summary = tk.StringVar(value="No tests run yet")
        ttk.Label(summary, textvariable=self.var_summary,
                  font=("Consolas", 10)).pack(side="left")

    def _on_select(self, _event):
        selection = self.tree.selection()
        if not selection:
            return
        test_id = selection[0]
        test = next(t for t in TESTS if t["id"] == test_id)

        self.var_test_name.set(test["name"])
        self.var_description.set(test["description"])
        self.btn_run_one.config(state="normal" if not self._running else "disabled")

        result = self._results.get(test_id)
        if result:
            self._show_output(result)
        else:
            self._set_output("Not run yet.", "info")

    def _show_output(self, result: Dict):
        tag = "pass" if result["passed"] else "fail"
        header = "PASSED" if result["passed"] else "FAILED"
        self._set_output(f"[ {header} ]\n\n{result['output']}", tag)

    def _set_output(self, text: str, tag: str = "info"):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", text, tag)
        self.output_text.config(state="disabled")

    def _update_tree(self, test_id: str, status: str):
        if status == "PASS":
            display = "PASS"
        elif status == "FAIL":
            display = "FAIL"
        elif status == "RUNNING":
            display = "..."
        else:
            display = status
        test = next(t for t in TESTS if t["id"] == test_id)
        self.tree.item(test_id, values=(display, test["name"]))

    def _update_summary(self):
        passed = sum(1 for r in self._results.values() if r and r["passed"])
        failed = sum(1 for r in self._results.values() if r and not r["passed"])
        total = passed + failed
        if total == 0:
            self.var_summary.set("No tests run yet")
        else:
            self.var_summary.set(f"{passed} passed, {failed} failed, {total} total")

    def _set_running(self, running: bool):
        self._running = running
        state = "disabled" if running else "normal"
        self.btn_run_selected.config(state=state)
        self.btn_run_one.config(state=state if self.tree.selection() else "disabled")

    # ── Run handlers ──────────────────────────────────────────────────

    def _run_all(self):
        if self._running:
            return
        test_ids = [t["id"] for t in TESTS]
        self._run_tests(test_ids)

    def _run_selected(self):
        if self._running:
            return
        selection = list(self.tree.selection())
        if not selection:
            return
        self._run_tests(selection)

    def _run_current(self):
        if self._running:
            return
        selection = list(self.tree.selection())
        if selection:
            self._run_tests([selection[0]])

    def _run_tests(self, test_ids: List[str]):
        self._set_running(True)
        self.progress["value"] = 0

        for tid in test_ids:
            self._update_tree(tid, "RUNNING")

        thread = threading.Thread(target=self._run_worker, args=(test_ids,), daemon=True)
        thread.start()

    def _run_worker(self, test_ids: List[str]):
        total = len(test_ids)
        for i, test_id in enumerate(test_ids):
            test = next(t for t in TESTS if t["id"] == test_id)
            self.after(0, self.var_status.set, f"Running: {test['name']}...")

            result = run_single_test(test["class"])
            self._results[test_id] = result

            status = "PASS" if result["passed"] else "FAIL"
            self.after(0, self._update_tree, test_id, status)
            self.after(0, self._update_summary)
            self.after(0, lambda v=(i + 1) / total * 100: self.progress.configure(value=v))

            # If this test is currently selected, update the detail panel
            self.after(0, self._refresh_detail_if_selected, test_id, result)

        self.after(0, self._on_batch_done)

    def _refresh_detail_if_selected(self, test_id: str, result: Dict):
        selection = self.tree.selection()
        if selection and selection[0] == test_id:
            self._show_output(result)

    def _on_batch_done(self):
        self._set_running(False)
        passed = sum(1 for r in self._results.values() if r and r["passed"])
        failed = sum(1 for r in self._results.values() if r and not r["passed"])
        self.var_status.set(f"Done: {passed} passed, {failed} failed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = TestRunnerApp()
    app.mainloop()
