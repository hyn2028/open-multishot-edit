"""Reads a data directory (`benchmark.jsonl`, `eval_metadata.json`, `source_videos/`)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    instruction_id: str
    task: str
    source_video: str
    edit_instruction: str
    shots: list[tuple[int, int]]  # [start, end) frame range of every shot
    # Text descriptions of the source / edited video, for methods conditioned on them instead of
    # an edit instruction.
    source_prompt: str
    target_prompt: str
    target_shot_prompts: list[str]  # one per shot


def load_scenarios(data_root: str) -> list[Scenario]:
    root = Path(data_root)
    meta = json.loads((root / "eval_metadata.json").read_text(encoding="utf-8"))
    scenarios = []
    for line in (root / "benchmark.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample = meta["samples"][row["sample_id"]]
        ins = meta["instructions"][row["instruction_id"]]
        scenarios.append(Scenario(
            instruction_id=row["instruction_id"],
            task=row["task"],
            source_video=str(root / row["source_video"]),
            edit_instruction=row["edit_instruction"],
            shots=[(int(s), int(e)) for s, e in sample["shots"]],
            source_prompt=sample["source_prompt"],
            target_prompt=ins["target_prompt"],
            target_shot_prompts=ins["target_shot_prompts"],
        ))
    return scenarios
