"""SDEdit with HoloCine, a multi-shot text-to-video model (Wan2.2-T2V-A14B based).

The source video is encoded, partially renoised (`--denoising-strength`), and denoised under
HoloCine's structured multi-shot prompt, built from the description of the edited video:

    [global caption] {target_prompt before "Shot sequence: "} This scene contains N shots.
    [per shot caption] {shot 1} [shot cut] {shot 2} [shot cut] ...

with shot cuts at the source's shot boundaries. This uses the unmodified `WanVideoHoloCinePipeline`
of the HoloCine repository (its `input_video` / `denoising_strength` arguments), with the full
attention checkpoint. Other sampling settings are the pipeline's defaults; the video is
processed at 832x480 and saved at the source fps.
"""

from __future__ import annotations

import os
import sys
import tempfile

from common import video

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "HoloCine"))  # diffsynth

WIDTH, HEIGHT = 832, 480

# Wan's default negative prompt, as used by HoloCine_inference_full_attention.py.
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# (repo, revision, files)
CHECKPOINTS = [
    ("Wan-AI/Wan2.2-T2V-A14B", "c8c270b13ee05bfa474194ac9fb07a5868a97cea", ["models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"]),
    ("hlwang06/HoloCine", "08ee5131abc3eb290c1f1fa93ba79a18fa63c289", ["HoloCine_dit/full/full_high_noise.safetensors", "HoloCine_dit/full/full_low_noise.safetensors"]),
    ("Wan-AI/Wan2.1-T2V-1.3B", "37ec512624d61f7aa208f7ea8140a131f93afc9a", ["google/umt5-xxl/*"]),  # tokenizer
]


def add_args(ap) -> None:
    ap.add_argument("--denoising-strength", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg-scale", type=float, default=5.0)
    ap.add_argument("--sigma-shift", type=float, default=5.0)
    ap.add_argument("--switch-dit-boundary", type=float, default=0.875)
    ap.add_argument("--seed", type=int, default=0)


def load(args):
    import torch
    from huggingface_hub import snapshot_download

    from diffsynth.pipelines.wan_video_holocine import WanVideoHoloCinePipeline
    from diffsynth.utils import ModelConfig

    wan, holocine, tokenizer = (snapshot_download(repo, revision=rev, allow_patterns=files) for repo, rev, files in CHECKPOINTS)
    pipe = WanVideoHoloCinePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=f"{wan}/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(path=f"{holocine}/HoloCine_dit/full/full_high_noise.safetensors", offload_device="cpu"),
            ModelConfig(path=f"{holocine}/HoloCine_dit/full/full_low_noise.safetensors", offload_device="cpu"),
            ModelConfig(path=f"{wan}/Wan2.1_VAE.pth", offload_device="cpu"),
        ],
        tokenizer_config=ModelConfig(path=f"{tokenizer}/google/umt5-xxl"),
    )
    pipe.enable_vram_management()
    pipe.to("cuda")
    return pipe


def multishot_prompt(global_caption: str, shot_captions: list[str]) -> str:
    return (f"[global caption] {global_caption} This scene contains {len(shot_captions)} shots. "
            f"[per shot caption] {' [shot cut] '.join(shot_captions)}")


def shot_cut_frames(shot_starts: list[int], num_frames: int) -> list[int]:
    """Each cut rounded to the nearest 4t+1 frame, as in HoloCine's own inference script."""
    cuts = {4 * round((f - 1) / 4) + 1 for f in shot_starts}
    return sorted(f for f in cuts if 0 < f < num_frames)


def edit(pipe, s, out_path: str, args) -> None:
    import imageio
    from PIL import Image

    from diffsynth import save_video

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "input.mp4")
        frames, padded = video.prepare_input(s.source_video, src, WIDTH, HEIGHT, stride=4, max_frames=args.max_frames)
        with imageio.get_reader(src) as reader:
            clip = [Image.fromarray(f) for f in reader]

        # Shots starting beyond the (shortened) video are dropped.
        shots = [(start, text) for (start, _), text in zip(s.shots, s.target_shot_prompts) if start < padded]
        global_caption = s.target_prompt.rsplit("Shot sequence: ", 1)[0].rstrip()
        out = pipe(
            prompt=multishot_prompt(global_caption, [text for _, text in shots]),
            negative_prompt=NEGATIVE_PROMPT,
            input_video=clip,
            denoising_strength=args.denoising_strength,
            shot_cut_frames=shot_cut_frames([start for start, _ in shots[1:]], padded),
            num_frames=padded,
            height=HEIGHT,
            width=WIDTH,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.steps,
            sigma_shift=args.sigma_shift,
            switch_DiT_boundary=args.switch_dit_boundary,
            seed=args.seed,
            tiled=True,
        )
        generated = os.path.join(tmp, "generated.mp4")
        save_video(out, generated, fps=video.fps(s.source_video), quality=5)
        video.trim(generated, out_path, frames)
