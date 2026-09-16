# Manipulations

Scripts that build manipulated variants of Anthropic/hh-rlhf, ready to feed
to `Training/baseline.py`. Three families of manipulation are supported:

- **REMOVAL** — filter out preference pairs whose CHOSEN response contains a
  sycophancy-adjacent phrase (from a hand-curated list) that REJECTED does not.
- **INSERTION** — append a JSONL of synthetic anti-sycophancy pairs to
  hh-rlhf, then shuffle. Three targeted pair sets + one combined set.
- **COMBINED** — removal followed by insertion, chained via a `+` in the
  method name.

## Entry point — `build_train_data.py`

Reads `Anthropic/hh-rlhf` from HuggingFace, applies a manipulation, writes
the result as parsed JSONL in the format `Training/baseline.py` consumes.

```powershell
# Baseline (no manipulation) — control corpus for training pipeline
python build_train_data.py method=none output_dir=./data/hh-rlhf-baseline

# Removal (5% of corpus filtered)
python build_train_data.py method=remove-naive-conservative `
                            output_dir=./data/hh-rlhf-v1a

# Insertion — one of four synthetic sets
python build_train_data.py method=insert-wei          output_dir=./data/hh-rlhf-v2a
python build_train_data.py method=insert-are-you-sure output_dir=./data/hh-rlhf-v2b
python build_train_data.py method=insert-answer       output_dir=./data/hh-rlhf-v2c
python build_train_data.py method=insert-combined     output_dir=./data/hh-rlhf-v2d

# Combined — removal + insertion (chained with `+`)
python build_train_data.py method=remove-naive-conservative+insert-are-you-sure `
                            output_dir=./data/hh-rlhf-v3a
python build_train_data.py method=remove-naive-conservative+insert-combined `
                            output_dir=./data/hh-rlhf-v3b
```

Each command produces `<output_dir>/train.jsonl` (manipulated training set)
and `<output_dir>/test.jsonl` (Anthropic/hh-rlhf test split, never manipulated).

## Variant naming used in the report

| Variant | Method | Description |
|---|---|---|
| baseline | none | Unmanipulated hh-rlhf |
| v1a | remove-naive-conservative | Filter ~4.6% of pairs by phrase list |
| v2a | insert-wei | Wei et al. (2024) feedback-sycophancy pairs |
| v2b | insert-are-you-sure | Arithmetic "are-you-sure" pairs |
| v2c | insert-answer | Common-misconceptions answer-format pairs |
| v2d | insert-combined | All three above, mixed one-third each (~5,000 total) |
| v3a | remove-naive-conservative + insert-are-you-sure | v1a filter, then v2b insertion |
| v3b | remove-naive-conservative + insert-combined | v1a filter, then v2d insertion |

## Building the synthetic insertion datasets from scratch (optional)

The three `insert_*.py` scripts generate the source JSONL files that
`build_train_data.py` consumes for the `insert-*` methods:

- `insert_Wei.py` → `Wei2024_Feedback_Sycophancy_rlhf.jsonl` (5,000 pairs)
- `insert_AreYouSure.py` → `AreYouSure_Sycophancy_rlhf.jsonl` (5,000 pairs)
- `insert_Answer.py` → `Answer_Sycophancy_rlhf.jsonl` (~6,800 pairs from
  ~570 Wikipedia misconceptions)
- `combine_insert_datasets.py` → `combined_Sycophancy_rlhf.jsonl` (5,000
  pairs, 1/3 from each of the three sets above)

These JSONLs are checked into the repo, so you only need to re-run the
`insert_*` scripts if you want to regenerate them from source data.

## Removal internals

- `remove_naive.py` — Saar's original 27-phrase list, case-sensitive
  (~0.9% removal).
- `remove_naive_expanded.py` — Shay's expanded 125-phrase list,
  case-insensitive (~4.6% removal in `conservative` mode, ~25% in
  `aggressive` mode).
- `phrase_list.md` — full inventory of phrases in both variants, with
  category breakdown and per-category chosen/rejected empirical rates.
