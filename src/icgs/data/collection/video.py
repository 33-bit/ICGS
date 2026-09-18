"""Deterministic video encoding utilities for episode archival and preview."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np


def encode_frames_to_mp4(
    frames: Sequence[np.ndarray] | np.ndarray,
    output_path: str | Path,
    fps: int = 10,
) -> bool:
    """Encode an RGB frame sequence to an MP4 video (H.264).

    Parameters
    ----------
    frames : Sequence[np.ndarray] | np.ndarray
        Sequence or array of shape (T, H, W, 3) with uint8 RGB images.
    output_path : str | Path
        Target MP4 destination path.
    fps : int
        Frames per second (default 10).

    Returns
    -------
    bool
        True if encoding succeeded and output file exists with positive size.
    """
    if len(frames) == 0:
        return False

    frames_arr = np.asarray(frames, dtype=np.uint8)
    if frames_arr.ndim != 4 or frames_arr.shape[-1] != 3:
        return False

    n, h, w, c = frames_arr.shape
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Strategy 1: ffmpeg CLI (fastest, standard H.264, universally available on Colab/Linux)
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is not None:
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{w}x{h}",
                "-pix_fmt", "rgb24",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-loglevel", "error",
                str(out_path),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            proc.communicate(input=frames_arr.tobytes())
            if proc.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 0:
                return True
        except Exception:
            pass

    # Strategy 2: OpenCV VideoWriter (if cv2 is installed)
    try:
        import cv2  # type: ignore
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
        for frame in frames_arr:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        if out_path.is_file() and out_path.stat().st_size > 0:
            return True
    except Exception:
        pass

    return False
