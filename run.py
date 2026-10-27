#!/usr/bin/env python3
"""Runs one editing method on every scenario of a data directory.

    python run.py <method> --data-root data --out-dir outputs/<method> [--gpus 0,1,2,3] [method options]

Writes {out-dir}/{instruction_id}.mp4 with the source's frame count, at the method's own
resolution (or --resize). Existing outputs are skipped, so an interrupted run can be resumed by
re-running the same command. Each GPU runs one worker process that loads the model once, using
the method's own environment (methods/<method>/.venv, created with `uv sync`).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
METHODS = sorted(d for d in os.listdir(os.path.join(ROOT, "methods")) if os.path.isfile(os.path.join(ROOT, "methods", d, "method.py")))


def parse_args() -> tuple[argparse.Namespace, object]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=METHODS)
    ap.add_argument("--data-root", required=True, help="data directory (benchmark.jsonl, eval_metadata.json, source_videos/)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids, one worker each (default: 0)")
    ap.add_argument("--ids", default=None, help="only run these comma-separated instruction ids")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N scenarios")
    ap.add_argument("--max-frames", type=int, default=None, help="only edit the first N frames of each source (for quick tests)")
    ap.add_argument("--resize", default=None, metavar="WxH", help="force-resize the outputs, e.g. 960x544")
    ap.add_argument("--rank", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--world-size", type=int, default=1, help=argparse.SUPPRESS)
    name = next((a for a in sys.argv[1:] if a in METHODS), None)
    method = None
    if name:
        method = importlib.import_module(f"methods.{name}.method")
        method.add_args(ap.add_argument_group(f"{name} options"))
    return ap.parse_args(), method


def launch(args: argparse.Namespace) -> int:
    python = os.path.join(ROOT, "methods", args.method, ".venv", "bin", "python")
    if not os.path.isfile(python):
        sys.exit(f"{python} not found; set up the environment first: (cd methods/{args.method} && uv sync)")
    gpus = args.gpus.split(",")
    procs = [
        subprocess.Popen(
            [python, os.path.abspath(__file__), *sys.argv[1:], "--rank", str(rank), "--world-size", str(len(gpus))],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
        )
        for rank, gpu in enumerate(gpus)
    ]
    return max(p.wait() for p in procs)


def work(args: argparse.Namespace, method) -> int:
    from common.data import load_scenarios
    from common import video

    scenarios = load_scenarios(args.data_root)
    if args.ids:
        ids = set(args.ids.split(","))
        scenarios = [s for s in scenarios if s.instruction_id in ids]
    scenarios = scenarios[:args.limit][args.rank::args.world_size]
    os.makedirs(args.out_dir, exist_ok=True)
    todo = [s for s in scenarios if not os.path.exists(os.path.join(args.out_dir, f"{s.instruction_id}.mp4"))]
    print(f"[{args.method} rank {args.rank}] {len(todo)} to run, {len(scenarios) - len(todo)} already done", flush=True)
    if not todo:
        return 0

    model = method.load(args)
    failed = 0
    for i, s in enumerate(todo):
        out = os.path.join(args.out_dir, f"{s.instruction_id}.mp4")
        tmp = os.path.join(args.out_dir, f".{s.instruction_id}.tmp.mp4")
        start = time.time()
        try:
            method.edit(model, s, tmp, args)
            if args.resize:
                width, height = map(int, args.resize.split("x"))
                resized = os.path.join(args.out_dir, f".{s.instruction_id}.resized.tmp.mp4")
                video.resize(tmp, resized, width, height)
                os.replace(resized, tmp)
            os.replace(tmp, out)
            record = {"instruction_id": s.instruction_id, "status": "ok"}
        except Exception as e:  # keep going with the other scenarios
            traceback.print_exc()
            failed += 1
            record = {"instruction_id": s.instruction_id, "status": "error", "error": repr(e)}
        record["seconds"] = round(time.time() - start, 1)
        print(f"[{args.method} rank {args.rank}] {i + 1}/{len(todo)} {s.instruction_id}: {record['status']} ({record['seconds']}s)", flush=True)
        with open(os.path.join(args.out_dir, "log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    return 1 if failed else 0


def main() -> int:
    args, method = parse_args()
    return launch(args) if args.rank is None else work(args, method)


if __name__ == "__main__":
    sys.exit(main())
