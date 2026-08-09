"""Apply a manipulation to hh-rlhf and save the result in the parsed JSONL
format that the training pipeline (Training/baseline.py) expects.

Bridge between Manipulations/ (Saar-style raw-string-format outputs) and
Training/ (structured-turn-list-format inputs).

Currently supports:
  method="remove-naive-conservative" — remove_naive_expanded conservative (~5%)
  method="remove-naive-aggressive"   — remove_naive_expanded aggressive (~25%)
  method="remove-naive-saar"         — original remove_naive.py (~1%)
  method="none"                      — no manipulation, produces the baseline

Output layout matches the training pipeline's ParsedHHRLHFBuilder:
  <output_dir>/train.jsonl   — filtered/manipulated training set
  <output_dir>/test.jsonl    — copied verbatim from Anthropic/hh-rlhf test split

Usage:
  python build_train_data.py method=remove-naive-conservative output_dir=./data/hh-rlhf-v1a-naive-conservative
  python build_train_data.py method=remove-naive-aggressive   output_dir=./data/hh-rlhf-v1b-naive-aggressive
  python build_train_data.py method=none                      output_dir=./data/hh-rlhf-baseline
"""
import json
import re
import sys
from pathlib import Path

import chz
from datasets import load_dataset


# --------------------------------------------------------------------------- #
# hh-rlhf conversation parser (duplicated from Training/preprocess.py so this
# module stands alone in Saar's repo without cross-imports)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Manipulation registry — maps method names to functions returning
# indices_to_remove given a HuggingFace Dataset.
# --------------------------------------------------------------------------- #
def _method_none(ds) -> set[int]:
    return set()


def _method_remove_naive_saar(ds) -> set[int]:
    # Import locally so this file works even if Saar's remove_naive.py is
    # missing (returns empty in that case).
    try:
        from remove_naive import naive_includes_sycophantic_phrase
    except ImportError:
        raise SystemExit("ERROR: remove_naive.py not found in current directory.")
    return set(naive_includes_sycophantic_phrase(ds))


def _method_remove_naive_expanded(ds, mode: str) -> set[int]:
    from remove_naive_expanded import naive_includes_sycophantic_phrase
    idxs, _stats = naive_includes_sycophantic_phrase(ds, mode=mode)
    return idxs


METHODS = {
    "none":                        lambda ds: _method_none(ds),
    "remove-naive-saar":           lambda ds: _method_remove_naive_saar(ds),
    "remove-naive-conservative":   lambda ds: _method_remove_naive_expanded(ds, "conservative"),
    "remove-naive-aggressive":     lambda ds: _method_remove_naive_expanded(ds, "aggressive"),
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@chz.chz
class Config:
    method: str = "remove-naive-conservative"
    output_dir: str = "./data/hh-rlhf-v1a-naive-conservative"


def main(cfg: Config) -> None:
    if cfg.method not in METHODS:
        raise SystemExit(f"Unknown method {cfg.method!r}. Options: {list(METHODS)}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method: {cfg.method}")
    print(f"Output: {out_dir.resolve()}")
    print()

    # ---- TRAIN split ----
    print("Loading hh-rlhf train split...")
    train_ds = load_dataset("Anthropic/hh-rlhf", split="train")
    total_train = len(train_ds)
    print(f"  {total_train:,} raw examples")

    print(f"Applying manipulation ({cfg.method})...")
    to_remove = METHODS[cfg.method](train_ds)
    kept_indices = [i for i in range(total_train) if i not in to_remove]
    print(f"  removing {len(to_remove):,} ({100*len(to_remove)/total_train:.2f}%)")
    print(f"  keeping  {len(kept_indices):,}")

    print("Parsing kept examples to structured JSONL...")
    train_path = out_dir / "train.jsonl"
    n_written = 0
    n_parse_failed = 0
    with open(train_path, "w", encoding="utf-8") as f:
        for i in kept_indices:
            parsed = parse_example(train_ds[i])
            if parsed is None:
                n_parse_failed += 1
                continue
            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"  wrote {n_written:,} to {train_path.name}  ({n_parse_failed} parse failures)")

    # ---- TEST split (verbatim from source) ----
    print("Loading hh-rlhf test split (unmanipulated)...")
    test_ds = load_dataset("Anthropic/hh-rlhf", split="test")
    total_test = len(test_ds)
    test_path = out_dir / "test.jsonl"
    n_test_ok = n_test_fail = 0
    with open(test_path, "w", encoding="utf-8") as f:
        for ex in test_ds:
            parsed = parse_example(ex)
            if parsed is None:
                n_test_fail += 1
                continue
            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            n_test_ok += 1
    print(f"  wrote {n_test_ok:,} to {test_path.name}  ({n_test_fail} parse failures)")

    print()
    print(f"Done. Point baseline.py at this folder with:")
    print(f"  python baseline.py data_dir={out_dir.resolve()} wandb_name=<pick-a-name>")


if __name__ == "__main__":
    main(chz.entrypoint(Config))
