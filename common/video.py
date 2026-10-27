"""ffmpeg/ffprobe helpers shared by all methods.

Sources are never cropped or truncated to fit a model: they are force-resized (aspect ratio
ignored) to the model's resolution and padded by repeating the last frame up to a valid frame
count; the output is trimmed back to the source length.
"""

from __future__ import annotations

import os
import subprocess


def num_frames(path: str) -> int:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        text=True,
    )
    return int(out.strip())


def fps(path: str) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
        text=True,
    )
    num, _, den = out.strip().partition("/")
    return float(num) / float(den) if den else float(num)


def prepare_input(src: str, dst: str, width: int, height: int, stride: int, max_frames: int | None = None) -> tuple[int, int]:
    """Writes `src` resized to width x height, cut to its first `max_frames` frames (if set), and
    padded so that (frames - 1) % stride == 0. Returns (frames to keep, padded frame count)."""
    total = num_frames(src)
    usable = min(total, max_frames) if max_frames is not None else total
    pad = -(usable - 1) % stride

    cut = None
    if usable < total:
        # Losslessly cut first; trimming with a filter can drop the last frame.
        cut = dst[:-len(".mp4")] + ".cut.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-frames:v", str(usable),
             "-an", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", cut],
            check=True,
        )
        src = cut
    filters = [f"scale={width}:{height}"]
    if pad:
        filters.append(f"tpad=stop_mode=clone:stop={pad}")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", ",".join(filters),
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", dst],
        check=True,
    )
    if cut:
        os.remove(cut)
    return usable, usable + pad


def trim(src: str, dst: str, frames: int) -> None:
    """Re-encodes the first `frames` frames of `src`."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-frames:v", str(frames),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4", dst],
        check=True,
    )


def resize(src: str, dst: str, width: int, height: int) -> None:
    """Force-resizes (aspect ratio ignored)."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", f"scale={width}:{height}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", "-f", "mp4", dst],
        check=True,
    )
