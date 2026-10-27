"""Lucy Edit (decart-ai/Lucy-Edit-Dev, diffusers `LucyEditPipeline`), conditioned on the edit
instruction. The whole video is edited in one pass at 832x480 (the model card's example
resolution); the frame count is padded to the model's 4k+1 requirement and trimmed back after.
Sampling settings follow the model card.
"""

from __future__ import annotations

import os
import tempfile

from common import video

MODEL_ID = "decart-ai/Lucy-Edit-Dev"
REVISION = "cb201fdec1bca6e7c362e127392c1632c92d2576"
WIDTH, HEIGHT = 832, 480


def add_args(ap) -> None:
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=24, help="output fps")


def load(args):
    import torch
    from diffusers import AutoencoderKLWan, LucyEditPipeline

    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", revision=REVISION, torch_dtype=torch.float32)
    pipe = LucyEditPipeline.from_pretrained(MODEL_ID, vae=vae, revision=REVISION, torch_dtype=torch.bfloat16)
    return pipe.to("cuda")


def edit(pipe, s, out_path: str, args) -> None:
    import torch
    from diffusers.utils import export_to_video, load_video

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "input.mp4")
        frames, padded = video.prepare_input(s.source_video, src, WIDTH, HEIGHT, stride=4, max_frames=args.max_frames)
        clip = load_video(src, convert_method=lambda v: [f.resize((WIDTH, HEIGHT)) for f in v[:padded]])

    out = pipe(
        prompt=s.edit_instruction,
        video=clip,
        negative_prompt=args.negative_prompt,
        height=HEIGHT,
        width=WIDTH,
        num_frames=padded,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    ).frames[0]
    export_to_video(out[:frames], out_path, fps=args.fps)
