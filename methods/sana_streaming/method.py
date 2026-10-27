"""SANA-Streaming, conditioned on the edit instruction.

Runs the unmodified `inference_video_scripts/v2v/inference_sana_streaming.py` of the Sana
repository in `long_streaming` mode (the only mode that covers a full-length video) with its
default sampling settings. The video is processed at 1280x704 (the script's default
resolution); the frame count is padded to the VAE's 8k+1 requirement and trimmed back after.
The script reloads the model for every video.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from common import video

SANA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sana")
WIDTH, HEIGHT = 1280, 704


def add_args(ap) -> None:
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=24, help="output fps")


def load(args):
    return None


def edit(_, s, out_path: str, args) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "input.mp4")
        frames, padded = video.prepare_input(s.source_video, src, WIDTH, HEIGHT, stride=8, max_frames=args.max_frames)
        subprocess.run(
            [sys.executable, "inference_video_scripts/v2v/inference_sana_streaming.py",
             "--mode", "long_streaming",
             "--video_path", src,
             "--prompt", s.edit_instruction,
             "--output_dir", tmp,
             "--output_name", "generated.mp4",
             "--num_frames", str(padded),
             "--height", str(HEIGHT), "--width", str(WIDTH),
             "--fps", str(args.fps),
             "--seed", str(args.seed)],
            cwd=SANA, check=True,
        )
        video.trim(os.path.join(tmp, "generated.mp4"), out_path, frames)
