# Open Multi-Shot Edit

Open-source video editing models, run on long multi-shot videos with one shared runner.

| Method | `run.py` name | Conditioned on | Code | Weights |
|---|---|---|---|---|
| Lucy Edit | `lucy_edit` | edit instruction | [diffusers](https://github.com/huggingface/diffusers) `LucyEditPipeline` | [decart-ai/Lucy-Edit-Dev](https://huggingface.co/decart-ai/Lucy-Edit-Dev) |
| SDEdit (HoloCine) | `holocine_sdedit` | edited video description (whole video and per shot), shot boundaries | [yihao-meng/HoloCine](https://github.com/yihao-meng/HoloCine) | [hlwang06/HoloCine](https://huggingface.co/hlwang06/HoloCine), [Wan-AI/Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) |
| SANA-Streaming | `sana_streaming` | edit instruction | [NVlabs/Sana](https://github.com/NVlabs/Sana) | downloaded by the Sana code |
| Wan-Edit (FlowAlign) | `wan_edit` | source video description, edited video description | [KyujinHan/Awesome-Training-Free-WAN2.1-Editing](https://github.com/KyujinHan/Awesome-Training-Free-WAN2.1-Editing) | [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) |

Besides the source video, the methods use the inputs listed above.

The upstream code is used unmodified, as git submodules, and model weights are downloaded at
fixed Hugging Face revisions. How each method is driven is described at the top of its
`methods/<name>/method.py`. The upstream code and weights keep their own licenses (see
[License](#license)).

## Setup

Each method has its own environment (the methods need incompatible versions of torch and
diffusers), locked with [uv](https://docs.astral.sh/uv/):

```bash
git clone --recursive https://github.com/hyn2028/open-multishot-edit.git
cd open-multishot-edit
(cd methods/lucy_edit && uv sync --frozen)
(cd methods/holocine_sdedit && uv sync --frozen)
(cd methods/sana_streaming && MAX_JOBS=8 uv sync --frozen)   # builds mmcv and flash-attn
(cd methods/wan_edit && uv sync --frozen)
```

`ffmpeg` and `ffprobe` must be on `PATH`. Model weights are downloaded to the Hugging Face cache
on first use. Building flash-attn for SANA-Streaming needs the CUDA toolkit (`nvcc`).

## Data

`--data-root` is a directory with:

```
data/
├── benchmark.jsonl        # one scenario per line
├── eval_metadata.json     # shots and text descriptions of each video
└── source_videos/
```

```jsonc
// benchmark.jsonl
{"instruction_id": "C1_4_A__remove__01", "sample_id": "C1_4_A", "source_video": "source_videos/....mp4",
 "task": "remove", "edit_instruction": "Remove the man"}

// eval_metadata.json (relevant fields)
{
  "samples": {
    "C1_4_A": {
      "shots": [[0, 96], [96, 188], [188, 291], [291, 361]],   // [start, end) frames
      "source_prompt": "<description of the source video>",
      "source_shot_prompts": ["<description of shot 1 of the source>", "..."]   // unused
    }
  },
  "instructions": {
    "C1_4_A__remove__01": {
      "target_prompt": "<description of the edited video, ending in 'Shot sequence: Shot 1: ... Shot 2: ...'>",
      "target_shot_prompts": ["<description of shot 1 after the edit>", "..."]
    }
  }
}
```

## Running

```bash
python run.py <method> --data-root data --out-dir outputs/<method> --gpus 0,1,2,3
```

Any Python 3.10+ can run `run.py`; it starts one worker per GPU in the method's environment. Each
worker loads the model once and edits its share of the scenarios. Outputs are written as
`{out-dir}/{instruction_id}.mp4`, with the same number of frames as the source, at the method's
own resolution (832×480, or 1280×704 for SANA-Streaming). Existing outputs are skipped, so
re-running the same command resumes an interrupted run; per-scenario status is appended to
`{out-dir}/log.jsonl`.

| Option | |
|---|---|
| `--gpus 0,1` | GPUs to use, one worker each (default: `0`) |
| `--ids a,b` / `--limit N` | run only some scenarios |
| `--max-frames N` | edit only the first N frames of each source, e.g. for a quick test |
| `--resize 960x544` | force-resize the outputs |

Method-specific options (seed, sampling steps, guidance, ...) are listed by
`python run.py <method> --help`.

Lucy Edit, Wan-Edit and SANA-Streaming edit a full video in one pass, so their memory use grows
with the number of frames; HoloCine offloads its weights to CPU memory. Cost of one 15 s video
(361 frames at 24 fps, 1280×720 source) on one RTX PRO 6000 Blackwell (96 GB), with the default
settings:

| Method | Time per video | Peak GPU memory | Peak CPU memory |
|---|---|---|---|
| Lucy Edit | 8 min | 39 GiB | 34 GiB |
| SDEdit (HoloCine) | 2 h 27 min | 79 GiB | 108 GiB |
| SANA-Streaming | 2 min | 42 GiB | 16 GiB |
| Wan-Edit (FlowAlign) | 35 min | 63 GiB | 34 GiB |

Time excludes model loading; memory is the peak `nvidia-smi` usage and process RSS.

## License

The code in this repository is released under the [MIT License](LICENSE). The upstream code (git
submodules) and the model weights are not part of this repository and keep their own licenses,
some of which are non-commercial.
