"""Apply a manipulation to hh-rlhf and save the result in the parsed JSONL
format that the training pipeline (Training/baseline.py) expects.

Bridge between Manipulations/ (Saar-style raw-string-format outputs) and
Training/ (structured-turn-list-format inputs).

Supports two families of manipulation:

  REMOVAL — filter out preference pairs whose CHOSEN response contains a
  sycophantic phrase and REJECTED does not:
    method="remove-naive-conservative" — remove_naive_expanded conservative (~5%)
    method="remove-naive-aggressive"   — remove_naive_expanded aggressive (~25%)
    method="remove-naive-saar"         — original remove_naive.py (~1%)
    method="none"                      — no manipulation, produces the baseline

  INSERTION (Saar's synthetic anti-sycophancy pairs) — append a JSONL of new
  pairs to hh-rlhf then shuffle:
    method="insert-wei"          — Wei2024_Feedback_Sycophancy_rlhf.jsonl
    method="insert-are-you-sure" — AreYouSure_Sycophancy_rlhf.jsonl
    method="insert-answer"       — Answer_Sycophancy_rlhf.jsonl
    method="insert-combined"     — combined_Sycophancy_rlhf.jsonl

Output layout matches the training pipeline's ParsedHHRLHFBuilder:
  <output_dir>/train.jsonl   — manipulated training set (parsed format)
  <output_dir>/test.jsonl    — Anthropic/hh-rlhf test split (parsed)

Usage:
  python build_train_data.py method=remove-naive-conservative output_dir=./data/hh-rlhf-v1a-naive-conservative
  python build_train_data.py method=insert-combined            output_dir=./data/hh-rlhf-v2d-insert-combined
  python build_train_data.py method=none                       output_dir=./data/hh-rlhf-baseline
"""
import json
import random
import re
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


def parse_example(example: dict) -> dict | None:
    """Parse a raw hh-rlhf-format {chosen, rejected} example to parsed format."""
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
# Removal helpers
# --------------------------------------------------------------------------- #
def _remove_naive_saar(ds) -> set[int]:
    try:
        from remove_naive import naive_includes_sycophantic_phrase
    except ImportError:
        raise SystemExit("ERROR: remove_naive.py not found in current directory.")
    return set(naive_includes_sycophantic_phrase(ds))


def _remove_naive_expanded(ds, mode: str) -> set[int]:
    from remove_naive_expanded import naive_includes_sycophantic_phrase
    idxs, _stats = naive_includes_sycophantic_phrase(ds, mode=mode)
    return idxs


# --------------------------------------------------------------------------- #
# Insertion helpers
# --------------------------------------------------------------------------- #
INSERT_FILES = {
    "insert-wei":          "Wei2024_Feedback_Sycophancy_rlhf.jsonl",
    "insert-are-you-sure": "AreYouSure_Sycophancy_rlhf.jsonl",
    "insert-answer":       "Answer_Sycophancy_rlhf.jsonl",
    "insert-combined":     "combined_Sycophancy_rlhf.jsonl",
}


def _load_insert_pairs(method: str) -> list[dict]:
    fname = INSERT_FILES[method]
    fpath = Path(fname)
    if not fpath.exists():
        raise SystemExit(f"ERROR: insert file {fname} not found in current directory.")
    with open(fpath, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    return pairs


# --------------------------------------------------------------------------- #
# Main pipeline — same interface regardless of method family
# --------------------------------------------------------------------------- #
def _apply_manipulation(method: str, train_ds) -> tuple[list[dict], dict]:
    """Return (list_of_parsed_train_records, stats_dict)."""
    stats = {"method": method}

    # Removal path
    if method.startswith("remove-") or method == "none":
        if method == "none":
            to_remove = set()
        elif method == "remove-naive-saar":
            to_remove = _remove_naive_saar(train_ds)
        elif method == "remove-naive-conservative":
            to_remove = _remove_naive_expanded(train_ds, "conservative")
        elif method == "remove-naive-aggressive":
            to_remove = _remove_naive_expanded(train_ds, "aggressive")
        else:
            raise SystemExit(f"Unknown removal method {method!r}")

        stats["n_removed"] = len(to_remove)
        stats["n_inserted"] = 0
        stats["removal_pct"] = 100 * len(to_remove) / len(train_ds)

        kept_records: list[dict] = []
        n_parse_failed = 0
        for i in range(len(train_ds)):
            if i in to_remove:
                continue
            parsed = parse_example(train_ds[i])
            if parsed is None:
                n_parse_failed += 1
                continue
            kept_records.append(parsed)
        stats["n_parse_failed"] = n_parse_failed
        return kept_records, stats

    # Insertion path
    if method.startswith("insert-"):
        # Parse the whole hh-rlhf training set
        base_records: list[dict] = []
        n_parse_failed = 0
        for ex in train_ds:
            parsed = parse_example(ex)
            if parsed is None:
                n_parse_failed += 1
                continue
            base_records.append(parsed)

        # Load and parse the insert JSONL (also in raw hh-rlhf {chosen, rejected} format)
        raw_inserts = _load_insert_pairs(method)
        insert_records: list[dict] = []
        n_insert_parse_failed = 0
        for ex in raw_inserts:
            parsed = parse_example(ex)
            if parsed is None:
                n_insert_parse_failed += 1
                continue
            insert_records.append(parsed)

        stats["n_removed"] = 0
        stats["n_inserted"] = len(insert_records)
        stats["n_parse_failed"] = n_parse_failed
        stats["n_insert_parse_failed"] = n_insert_parse_failed

        combined = base_records + insert_records
        rng = random.Random(42)
        rng.shuffle(combined)
        return combined, stats

    raise SystemExit(f"Unknown method {method!r}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@chz.chz
class Config:
    method: str = "remove-naive-conservative"
    output_dir: str = "./data/hh-rlhf-v1a-naive-conservative"


def main(cfg: Config) -> None:
    known_methods = ["none",
                     "remove-naive-saar", "remove-naive-conservative", "remove-naive-aggressive",
                     "insert-wei", "insert-are-you-sure", "insert-answer", "insert-combined"]
    if cfg.method not in known_methods:
        raise SystemExit(f"Unknown method {cfg.method!r}. Options: {known_methods}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method: {cfg.method}")
    print(f"Output: {out_dir.resolve()}")
    print()

    # ---- TRAIN split ----
    print("Loading hh-rlhf train split...")
    train_ds = load_dataset("Anthropic/hh-rlhf", split="train")
    print(f"  {len(train_ds):,} raw examples")

    print(f"Applying manipulation ({cfg.method})...")
    records, stats = _apply_manipulation(cfg.method, train_ds)
    print(f"  removed:  {stats['n_removed']:,}")
    print(f"  inserted: {stats['n_inserted']:,}")
    print(f"  parse failures (hh-rlhf): {stats['n_parse_failed']}")
    if "n_insert_parse_failed" in stats:
        print(f"  parse failures (insert):  {stats['n_insert_parse_failed']}")

    train_path = out_dir / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records):,} rows to {train_path.name}")

    # ---- TEST split (verbatim from source, never manipulated) ----
    print("Loading hh-rlhf test split (unmanipulated)...")
    test_ds = load_dataset("Anthropic/hh-rlhf", split="test")
    test_path = out_dir / "test.jsonl"
    n_ok = n_fail = 0
    with open(test_path, "w", encoding="utf-8") as f:
        for ex in test_ds:
            parsed = parse_example(ex)
            if parsed is None:
                n_fail += 1
                continue
            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"  wrote {n_ok:,} rows to {test_path.name}  ({n_fail} parse failures)")

    print()
    print(f"Done. Point baseline.py at this folder with:")
    print(f"  python baseline.py data_dir={out_dir.resolve()} wandb_name=<pick-a-name> kl_penalty_coef=0.2")


if __name__ == "__main__":
    main(chz.entrypoint(Config))
