"""Wan-Edit: training-free FlowAlign editing on Wan2.1-T2V-1.3B, from
Awesome-Training-Free-WAN2.1-Editing.

FlowAlign edits from a description of the source video to a description of the edited video
(`source_prompt` -> `target_prompt`), using the repository's diffusers fork
(`WanPipeline.flowalign`) and its MasaCtrl attention processor. Settings follow the
repository's example configs, without attention masking. The video is processed at 832x480;
the frame count is padded to the VAE's 4k+1 requirement and trimmed back after, and the output
is saved at the source fps. `flowalign` draws part of its noise from the global random state,
which the repository leaves unseeded; it is seeded with `--seed` before every video so that runs
are reproducible.
"""

from __future__ import annotations

import os
import sys
import tempfile

from common import video

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Awesome-Training-Free-WAN2.1-Editing", "utils"))  # wan_attention

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
WIDTH, HEIGHT = 832, 480
TOTAL_LAYERS = 30  # Wan2.1-T2V-1.3B


def add_args(ap) -> None:
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--target-guidance-scale", type=float, default=19.5)
    ap.add_argument("--zeta-scale", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)


def load(args):
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", revision=REVISION, torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, revision=REVISION, torch_dtype=torch.float16)
    # flow_shift 3.0 for 480P
    pipe.scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=3.0)
    return pipe.to("cuda")


def edit(pipe, s, out_path: str, args) -> None:
    import torch
    from diffusers.utils import export_to_video, load_video
    from wan_attention import register_attention_processor

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "input.mp4")
        frames, padded = video.prepare_input(s.source_video, src, WIDTH, HEIGHT, stride=4, max_frames=args.max_frames)
        clip = load_video(src)

    total_steps = int(args.steps * args.strength)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # the processor creates an (unused) ./results/mask/ directory
        try:
            register_attention_processor(
                pipe.transformer, processor_type="MasaCtrlProcessor",
                start_step=1, end_step=total_steps, start_layer=1, end_layer=TOTAL_LAYERS, layer_idx=None,
                total_layers=TOTAL_LAYERS, total_steps=total_steps, masking=False, masking_layer=list(range(11, 18)),
                frame_num=padded,
            )
        finally:
            os.chdir(cwd)
    torch.manual_seed(args.seed)
    out = pipe.flowalign(
        video=clip,
        source_prompt=s.source_prompt,
        target_prompt=s.target_prompt,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=args.steps,
        strength=args.strength,
        target_guidance_scale=args.target_guidance_scale,
        fg_zeta_scale=args.zeta_scale,
        bg_zeta_scale=args.zeta_scale,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    ).frames[0]
    export_to_video(out[:frames], out_path, fps=video.fps(s.source_video))
