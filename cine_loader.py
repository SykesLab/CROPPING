"""
CINE file loader for calibration z-stacks.

Handles loading .cine files for z-stack calibration data.
Uses pyphantom library (Phantom SDK) for .cine file access.

Usage:
    loader = CineLoader(cine_path)
    if loader.is_available():
        frames, positions = loader.extract_zstack(z_start=-6, z_end=6)
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import cv2


# =============================================================================
# Silent pyphantom import (suppress SDK banner)
# =============================================================================
@contextmanager
def _suppress_output() -> Generator[None, None, None]:
    """Suppress stdout/stderr at OS level for C extensions."""
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)

    try:
        yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)


# Try to import pyphantom
_cine_module: Optional[Any] = None
_utils_module: Optional[Any] = None
PYPHANTOM_AVAILABLE = False

try:
    with _suppress_output():
        from pyphantom import cine as _cine_module
        from pyphantom import utils as _utils_module
        PYPHANTOM_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# CineLoader Class
# =============================================================================
class CineLoader:
    """
    Load and extract frames from .cine files for z-stack calibration.

    Attributes:
        path: Path to .cine file
        cine_obj: pyphantom Cine object (if loaded)
        frame_range: Tuple of (first_frame, last_frame)
        num_frames: Total number of frames
    """

    def __init__(self, cine_path: Optional[str] = None):
        """
        Initialize loader, optionally loading a .cine file.

        Args:
            cine_path: Path to .cine file (optional, can load later)
        """
        self.path: Optional[Path] = None
        self.cine_obj: Optional[Any] = None
        self.frame_range: Tuple[int, int] = (0, 0)
        self.num_frames: int = 0
        self._image_shape: Optional[Tuple[int, int]] = None

        if cine_path:
            self.load(cine_path)

    @staticmethod
    def is_available() -> bool:
        """Check if pyphantom is available."""
        return PYPHANTOM_AVAILABLE

    def load(self, cine_path: str) -> bool:
        """
        Load a .cine file.

        Args:
            cine_path: Path to .cine file

        Returns:
            True if loaded successfully, False otherwise
        """
        if not PYPHANTOM_AVAILABLE:
            print("[CineLoader] pyphantom not available - install Phantom SDK")
            return False

        self.path = Path(cine_path)
        if not self.path.exists():
            print(f"[CineLoader] File not found: {cine_path}")
            return False

        try:
            self.cine_obj = _cine_module.Cine.from_filepath(str(self.path))
            if self.cine_obj is None:
                print(f"[CineLoader] Failed to open: {cine_path}")
                return False

            # Get frame range
            self.frame_range = self.cine_obj.range
            self.num_frames = self.frame_range[1] - self.frame_range[0] + 1

            # Get image shape from first frame
            first_frame = self._load_frame(self.frame_range[0])
            if first_frame is not None:
                self._image_shape = first_frame.shape

            return True

        except Exception as e:
            print(f"[CineLoader] Error loading {cine_path}: {e}")
            return False

    def _load_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Load a single frame as grayscale uint8."""
        if self.cine_obj is None or _utils_module is None:
            return None

        try:
            frame_range = _utils_module.FrameRange(frame_idx, frame_idx)
            frame = self.cine_obj.get_images(frame_range, Option=1)
            arr = np.squeeze(frame).astype(np.float32)

            # Convert to grayscale if needed
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)

            # Normalize to 0-255
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
            return arr.astype(np.uint8)

        except Exception as e:
            print(f"[CineLoader] Error loading frame {frame_idx}: {e}")
            return None

    def get_info(self) -> Dict[str, Any]:
        """Get information about the loaded .cine file."""
        if self.cine_obj is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "path": str(self.path) if self.path else None,
            "filename": self.path.name if self.path else None,
            "first_frame": self.frame_range[0],
            "last_frame": self.frame_range[1],
            "num_frames": self.num_frames,
            "image_width": self._image_shape[1] if self._image_shape else 0,
            "image_height": self._image_shape[0] if self._image_shape else 0,
        }

    def extract_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Extract a single frame by index.

        Args:
            frame_idx: Frame index (within cine's frame range)

        Returns:
            Grayscale uint8 numpy array, or None if failed
        """
        return self._load_frame(frame_idx)

    def extract_zstack(
        self,
        z_start: float,
        z_end: float,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
        frame_step: int = 1,
        average_frames: int = 1
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract z-stack frames with computed z-positions.

        The z-positions are linearly interpolated between z_start and z_end
        based on frame indices.

        Args:
            z_start: Z position (mm) at first frame (defocus, can be negative)
            z_end: Z position (mm) at last frame
            frame_start: First frame to extract (default: first in cine)
            frame_end: Last frame to extract (default: last in cine)
            frame_step: Frame increment (default: 1 = every frame)
            average_frames: Number of frames to average at each position (default: 1)

        Returns:
            Tuple of (images list, z_positions list)
        """
        if self.cine_obj is None:
            return [], []

        # Default to full range
        if frame_start is None:
            frame_start = self.frame_range[0]
        if frame_end is None:
            frame_end = self.frame_range[1]

        # Clamp to valid range
        frame_start = max(frame_start, self.frame_range[0])
        frame_end = min(frame_end, self.frame_range[1])

        # Generate frame indices
        frame_indices = list(range(frame_start, frame_end + 1, frame_step))
        n_positions = len(frame_indices)

        if n_positions == 0:
            return [], []

        # Compute z-positions (linear interpolation)
        z_positions = []
        for i, _ in enumerate(frame_indices):
            if n_positions == 1:
                z = (z_start + z_end) / 2
            else:
                t = i / (n_positions - 1)
                z = z_start + t * (z_end - z_start)
            z_positions.append(z)

        # Extract frames
        images = []
        for frame_idx in frame_indices:
            if average_frames > 1:
                # Average multiple frames
                frames_to_avg = []
                for offset in range(average_frames):
                    actual_idx = frame_idx + offset
                    if actual_idx <= self.frame_range[1]:
                        f = self._load_frame(actual_idx)
                        if f is not None:
                            frames_to_avg.append(f.astype(np.float32))

                if frames_to_avg:
                    avg = np.mean(frames_to_avg, axis=0).astype(np.uint8)
                    images.append(avg)
                else:
                    images.append(None)
            else:
                images.append(self._load_frame(frame_idx))

        # Filter out None values (keeping positions aligned)
        valid_images = []
        valid_positions = []
        for img, z in zip(images, z_positions):
            if img is not None:
                valid_images.append(img)
                valid_positions.append(z)

        return valid_images, valid_positions

    def extract_at_positions(
        self,
        stage_positions: List[float],
        stage_to_defocus_offset: float = 0.0,
        frame_indices: Optional[List[int]] = None
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract frames at specific stage positions.

        Use this when you have explicit stage position values (e.g., from CSV).

        Args:
            stage_positions: List of stage positions in mm
            stage_to_defocus_offset: Offset to convert stage to defocus (z = stage - offset)
            frame_indices: Corresponding frame indices (default: 0, 1, 2, ...)

        Returns:
            Tuple of (images list, defocus_positions list)
        """
        if self.cine_obj is None:
            return [], []

        if frame_indices is None:
            frame_indices = list(range(len(stage_positions)))

        images = []
        defocus_positions = []

        for stage_pos, frame_idx in zip(stage_positions, frame_indices):
            actual_idx = self.frame_range[0] + frame_idx
            if self.frame_range[0] <= actual_idx <= self.frame_range[1]:
                img = self._load_frame(actual_idx)
                if img is not None:
                    images.append(img)
                    defocus_positions.append(stage_pos - stage_to_defocus_offset)

        return images, defocus_positions


# =============================================================================
# Utility Functions
# =============================================================================
def check_pyphantom() -> Tuple[bool, str]:
    """
    Check if pyphantom is available and return status message.

    Returns:
        Tuple of (is_available, status_message)
    """
    if PYPHANTOM_AVAILABLE:
        return True, "pyphantom available - .cine files supported"
    else:
        return False, "pyphantom not installed - install Phantom SDK for .cine support"


def get_cine_info(cine_path: str) -> Optional[Dict[str, Any]]:
    """
    Quick utility to get info about a .cine file without keeping it loaded.

    Args:
        cine_path: Path to .cine file

    Returns:
        Info dict or None if failed
    """
    loader = CineLoader(cine_path)
    if loader.cine_obj is not None:
        return loader.get_info()
    return None
