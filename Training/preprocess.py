"""One-time preprocessing: parse Anthropic/hh-rlhf into structured JSONL.

Each output line is a JSON object with:
  - prompt:   list of {"role": "user"|"assistant", "content": str}
  - chosen:   str (the preferred final assistant response)
  - rejected: str (the dispreferred final assistant response)

Downstream experiments (baseline, manipulations) read from these files
instead of re-parsing the raw dataset every time.
"""
import json
import os
import re
from pathlib import Path

import chz
import datasets


_TURN_PATTERN = re.compile(r"\n\n(Human|Assistant): ")


def parse_conversation(text: str) -> list[dict[str, str]] | None:
    parts = _TURN_PATTERN.split(text)
    if len(parts) < 3:
        return None
    turns: list[dict[str, str]] = []
    i = 1
    while i + 1 < len(parts):
        role = "user" if parts[i] == "Human" else "assistant"
        content = parts[i + 1].strip()
        if not content:
            return None
        turns.append({"role": role, "content": content})
        i += 2
    return turns


def parse_example(example: dict[str, str]) -> dict | None:
    chosen_turns = parse_conversation(example["chosen"])
    rejected_turns = parse_conversation(example["rejected"])
    if not chosen_turns or not rejected_turns:
        return None
    if chosen_turns[-1]["role"] != "assistant" or rejected_turns[-1]["role"] != "assistant":
        return None
    prompt = chosen_turns[:-1]
    if not prompt:
        return None
    return {
        "prompt": prompt,
        "chosen": chosen_turns[-1]["content"],
        "rejected": rejected_turns[-1]["content"],
    }


def preprocess_split(split: str, output_path: Path) -> tuple[int, int]:
    ds = datasets.load_dataset("Anthropic/hh-rlhf", split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_dropped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in ds:
            parsed = parse_example(ex)
            if parsed is None:
                n_dropped += 1
            else:
                f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                n_ok += 1
    return n_ok, n_dropped


@chz.chz
class Config:
    output_dir: str = "./data/hh-rlhf-parsed"


def main(cfg: Config) -> None:
    out_dir = Path(cfg.output_dir)
    for split in ["train", "test"]:
        out_path = out_dir / f"{split}.jsonl"
        print(f"Parsing {split} -> {out_path}")
        n_ok, n_dropped = preprocess_split(split, out_path)
        print(f"  {n_ok} ok, {n_dropped} dropped")
    print(f"\nDone. Files at: {out_dir.resolve()}")


if __name__ == "__main__":
    main(chz.entrypoint(Config))
