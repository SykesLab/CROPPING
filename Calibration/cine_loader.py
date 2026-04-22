"""
CINE file loader for calibration z-stacks.

Handles loading .cine files for z-stack calibration data.
Supports both:
  - Single .cine file with multiple frames (one per z-position)
  - Folder of .cine files (one file per z-position)

Uses pyphantom library (Phantom SDK) for .cine file access.

Usage:
    # For folder of .cine files:
    loader = CineFolderLoader(folder_path)
    images, positions = loader.load_zstack(z_start=-6, z_end=6)

    # For single .cine file:
    loader = CineLoader(cine_path)
    images, positions = loader.extract_zstack(z_start=-6, z_end=6)
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Import shared phantom utilities from Preprocessing
_preproc_dir = str(Path(__file__).resolve().parent.parent / "Preprocessing")
if _preproc_dir not in sys.path:
    sys.path.insert(0, _preproc_dir)

from phantom_utils import init_pyphantom

_cine_module, _utils_module, _phantom_instance, PYPHANTOM_AVAILABLE, _ = init_pyphantom()


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
        """Load a single frame as grayscale float32 (raw values, no normalization)."""
        if self.cine_obj is None or _utils_module is None:
            return None

        try:
            frame_range = _utils_module.FrameRange(frame_idx, frame_idx)
            frame = self.cine_obj.get_images(frame_range, Option=1)
            arr = np.squeeze(frame).astype(np.float32)

            # Convert to grayscale if needed
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

            return arr

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
                    avg = np.mean(frames_to_avg, axis=0).astype(np.float32)
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
# CineFolderLoader Class - For folder of .cine files
# =============================================================================
class CineFolderLoader:
    """
    Load z-stack from a folder of .cine files (one file per z-position).

    Each .cine file represents one z-position. The loader extracts one frame
    (or averages multiple frames) from each file.

    Attributes:
        folder: Path to folder containing .cine files
        cine_files: List of .cine file paths (sorted)
        num_files: Number of .cine files found
    """

    def __init__(self, folder_path: Optional[str] = None):
        """
        Initialize loader, optionally scanning a folder.

        Args:
            folder_path: Path to folder containing .cine files
        """
        self.folder: Optional[Path] = None
        self.cine_files: List[Path] = []
        self.num_files: int = 0
        self._image_shape: Optional[Tuple[int, int]] = None

        if folder_path:
            self.scan_folder(folder_path)

    @staticmethod
    def is_available() -> bool:
        """Check if pyphantom is available."""
        return PYPHANTOM_AVAILABLE

    def scan_folder(self, folder_path: str) -> bool:
        """
        Scan a folder for .cine files.

        Args:
            folder_path: Path to folder

        Returns:
            True if .cine files found, False otherwise
        """
        self.folder = Path(folder_path)
        if not self.folder.exists():
            print(f"[CineFolderLoader] Folder not found: {folder_path}")
            return False

        # Find all .cine files and sort them
        self.cine_files = sorted(self.folder.glob("*.cine"))
        self.num_files = len(self.cine_files)

        if self.num_files == 0:
            print(f"[CineFolderLoader] No .cine files found in: {folder_path}")
            return False

        # Get image shape from first file
        if PYPHANTOM_AVAILABLE:
            loader = CineLoader(str(self.cine_files[0]))
            if loader.cine_obj is not None:
                info = loader.get_info()
                self._image_shape = (info['image_height'], info['image_width'])

        return True

    def get_info(self) -> Dict[str, Any]:
        """Get information about the folder."""
        return {
            "folder": str(self.folder) if self.folder else None,
            "num_files": self.num_files,
            "filenames": [f.name for f in self.cine_files],
            "image_width": self._image_shape[1] if self._image_shape else 0,
            "image_height": self._image_shape[0] if self._image_shape else 0,
        }

    def _extract_frame_from_cine(
        self,
        cine_path: Path,
        frame_idx: int = 0,
        average_frames: int = 1
    ) -> Optional[np.ndarray]:
        """Extract a frame (or averaged frames) from a single .cine file."""
        if not PYPHANTOM_AVAILABLE:
            return None

        loader = CineLoader(str(cine_path))
        if loader.cine_obj is None:
            return None

        if average_frames <= 1:
            return loader.extract_frame(loader.frame_range[0] + frame_idx)
        else:
            # Average multiple frames
            frames = []
            for i in range(average_frames):
                idx = loader.frame_range[0] + frame_idx + i
                if idx <= loader.frame_range[1]:
                    f = loader.extract_frame(idx)
                    if f is not None:
                        frames.append(f.astype(np.float32))

            if frames:
                return np.mean(frames, axis=0).astype(np.float32)
            return None

    def load_zstack(
        self,
        z_start: float,
        z_end: float,
        frame_idx: int = 0,
        average_frames: int = 1,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[np.ndarray], List[float], List[str]]:
        """
        Load z-stack from folder of .cine files.

        Args:
            z_start: Z position (mm) for first file (defocus)
            z_end: Z position (mm) for last file
            frame_idx: Which frame to extract from each .cine (default: 0 = first)
            average_frames: Number of frames to average per file (default: 1)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Tuple of (images list, z_positions list, filenames list)
        """
        if not PYPHANTOM_AVAILABLE:
            print("[CineFolderLoader] pyphantom not available")
            return [], [], []

        if self.num_files == 0:
            return [], [], []

        # Compute z-positions (linear interpolation based on file order)
        z_positions = []
        for i in range(self.num_files):
            if self.num_files == 1:
                z = (z_start + z_end) / 2
            else:
                t = i / (self.num_files - 1)
                z = z_start + t * (z_end - z_start)
            z_positions.append(z)

        # Extract one frame from each .cine file
        images = []
        filenames = []
        valid_positions = []

        for i, cine_path in enumerate(self.cine_files):
            if progress_callback:
                progress_callback(i + 1, self.num_files)

            img = self._extract_frame_from_cine(cine_path, frame_idx, average_frames)
            if img is not None:
                images.append(img)
                filenames.append(cine_path.name)
                valid_positions.append(z_positions[i])

        return images, valid_positions, filenames

    def load_with_positions_csv(
        self,
        csv_path: str,
        stage_offset: float = 0.0,
        frame_idx: int = 0,
        average_frames: int = 1
    ) -> Tuple[List[np.ndarray], List[float], List[str]]:
        """
        Load z-stack using a CSV file for position mapping.

        CSV should have columns: filename, z_position_mm (or stage_position_mm)

        Args:
            csv_path: Path to CSV file
            stage_offset: Offset to convert stage to defocus (z = stage - offset)
            frame_idx: Which frame to extract from each .cine
            average_frames: Number of frames to average

        Returns:
            Tuple of (images list, z_positions list, filenames list)
        """
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Use first column as filename, second as position (ignore column names)
        filenames_col = df.iloc[:, 0].astype(str)
        positions_col = df.iloc[:, 1].astype(float)

        # Build lookup dict (with and without extension)
        pos_dict = {}
        for fn, pos in zip(filenames_col, positions_col):
            pos_dict[fn] = pos
            pos_dict[Path(fn).stem] = pos

        images = []
        positions = []
        filenames = []

        for cine_path in self.cine_files:
            # Try to find position in CSV
            position = None
            if cine_path.name in pos_dict:
                position = pos_dict[cine_path.name] - stage_offset
            elif cine_path.stem in pos_dict:
                position = pos_dict[cine_path.stem] - stage_offset

            if position is not None:
                img = self._extract_frame_from_cine(cine_path, frame_idx, average_frames)
                if img is not None:
                    images.append(img)
                    positions.append(position)
                    filenames.append(cine_path.name)

        return images, positions, filenames

    def load_with_parsed_positions(
        self,
        focus_offset: float = 6.0,
        frame_idx: int = 0,
        average_frames: int = 1,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[np.ndarray], List[float], List[str]]:
        """
        Load z-stack by parsing stage positions from filenames.

        Uses naming convention: {prefix}_{run}_{index}.cine
        Where: stage_position = (run - 1) * 2 + (index - 1) * 0.2

        Args:
            focus_offset: Stage position where focus is (defocus = stage - offset)
            frame_idx: Which frame to extract from each .cine (default: 0)
            average_frames: Number of frames to average per file (default: 1)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Tuple of (images list, defocus_positions list, filenames list)
        """
        if not PYPHANTOM_AVAILABLE:
            print("[CineFolderLoader] pyphantom not available")
            return [], [], []

        if self.num_files == 0:
            return [], [], []

        # Parse positions from filenames and pair with files
        file_positions = []
        for cine_path in self.cine_files:
            stage_pos = parse_position_from_filename(cine_path.name)
            if stage_pos is not None:
                defocus = stage_pos - focus_offset
                file_positions.append((cine_path, defocus, cine_path.name))

        # Sort by defocus position
        file_positions.sort(key=lambda x: x[1])

        # Extract frames
        images = []
        positions = []
        filenames = []

        for i, (cine_path, defocus, name) in enumerate(file_positions):
            if progress_callback:
                progress_callback(i + 1, len(file_positions))

            img = self._extract_frame_from_cine(cine_path, frame_idx, average_frames)
            if img is not None:
                images.append(img)
                positions.append(defocus)
                filenames.append(name)

        return images, positions, filenames

    def get_parsed_positions_info(self, focus_offset: float = 6.0) -> Dict[str, Any]:
        """
        Get info about positions parsed from filenames (without loading images).

        Args:
            focus_offset: Stage position where focus is

        Returns:
            Dict with parsed position info
        """
        parsed = []
        unparsed = []

        for cine_path in self.cine_files:
            stage_pos = parse_position_from_filename(cine_path.name)
            if stage_pos is not None:
                defocus = stage_pos - focus_offset
                parsed.append((cine_path.name, stage_pos, defocus))
            else:
                unparsed.append(cine_path.name)

        # Sort by defocus
        parsed.sort(key=lambda x: x[2])

        if parsed:
            defocus_values = [p[2] for p in parsed]
            return {
                "num_parsed": len(parsed),
                "num_unparsed": len(unparsed),
                "unparsed_files": unparsed,
                "defocus_min": min(defocus_values),
                "defocus_max": max(defocus_values),
                "defocus_range": max(defocus_values) - min(defocus_values),
                "positions": parsed,  # List of (filename, stage_pos, defocus)
            }
        else:
            return {
                "num_parsed": 0,
                "num_unparsed": len(unparsed),
                "unparsed_files": unparsed,
            }


# =============================================================================
# Filename Position Parsing
# =============================================================================
def parse_position_from_filename(filename: str) -> Optional[float]:
    """
    Parse stage position from filename using naming convention.

    Supports format: {prefix}_{run}_{index}.cine
    Where: stage_position = (run - 1) * 2 + (index - 1) * 0.2

    Examples:
        9mm_1_1.cine  → 0.0 mm
        9mm_1_10.cine → 1.8 mm
        9mm_2_1.cine  → 2.0 mm
        9mm_4_5.cine  → 6.8 mm

    Args:
        filename: Filename (with or without path)

    Returns:
        Stage position in mm, or None if pattern doesn't match
    """
    # Extract just the filename without path
    name = Path(filename).stem

    # Pattern: {prefix}_{run}_{index}
    match = re.search(r'_(\d+)_(\d+)$', name)
    if match:
        run = int(match.group(1))
        index = int(match.group(2))
        # stage_position = (run - 1) * 2 + (index - 1) * 0.2
        position = (run - 1) * 2.0 + (index - 1) * 0.2
        return position

    return None


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
