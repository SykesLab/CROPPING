"""Standalone diagnostic tool: blur / domain fingerprint checker.

Verifies that the four image-producing/-consuming pipelines in CROPPING
(image preprocessing, synthetic generation, calibration, inference
preprocessing) produce equivalent images and that the scale-conversion
arithmetic round-trips cleanly. Read-only — does not modify any project
data or configs.

See ``../../.planning`` and the plan file at ~/.claude/plans/ for the
design rationale.
"""
