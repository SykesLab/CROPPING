"""Modular widgets for the Inference GUI.

Each widget is a ttk.Frame subclass with a clear public API:
``__init__(parent, on_event=...)`` for setup and ``update_state(...)``
for refresh. Widgets do not own application state — the main
``InferenceApp`` does — they emit events that the App responds to.
"""
