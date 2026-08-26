# `old/` folders — superseded / debugging leftovers

Files moved out of the active results tree because they are no longer canonical.
Kept for provenance in case anyone needs to trace an earlier number.

## What lives in each `old/` folder

**`results/old/`** — root
- `all_results.xlsx` + `build_results_xlsx.py`: the first summary XLSX I built, based on the keyword-graded numbers. Superseded by `results/all_raw_data.xlsx` + the LLM-regraded summary CSVs.

**`results/answer/old/`**
- `kl20-mini.*`: leftover from the KL-penalty tuning phase (a 23-batch mini-run at `kl_penalty_coef=0.2`). Not part of the 8-variant comparison.
- `v0-baseline-v2.*` (no `-sharma-methodology` suffix): the older baseline sample from before we ran the "sharma-methodology" batch. Different sampling seed; `v0-baseline-v2-sharma-methodology.jsonl` is the canonical baseline used everywhere in the report.
- `<variant>_summary.csv`: keyword-lexicon-graded summaries. Superseded by `<variant>_llm_summary.csv` (GPT-4o-mini semantic grading) — see the "LLM regrade" note below.

**`results/are_you_sure/old/`** and **`results/are_you_sure_pushed/old/`**
- `kl20-mini.*`: KL-tuning leftover, same as above.
- `v0-baseline-v2.*` (no suffix): produced by the BUGGY 2-turn-all-prefilled `are_you_sure` evaluator. Reported ~0% capitulation on models Sharma reported ~78% for. Rewritten to Sharma's 4-turn structure — see `v0-baseline-v2-sharma-methodology.jsonl` for the correct baseline. The old file is kept only as evidence of the pre-fix behavior.

**`results/feedback/old/`**
- `v0-baseline-v2.*` (no suffix): older sampling of the feedback eval on the baseline. Same script as `v0-baseline-v2-sharma-methodology.jsonl` (feedback methodology never changed), just a different sample seed.
- `<variant>_summary.csv`: keyword-lexicon-graded summaries. Superseded by `<variant>_llm_summary.csv` (GPT-4o-mini 1-5 endorse scoring).

## Why the keyword summaries were superseded

The original `answer` grader counted a response as `correct` if the correct-answer string appeared anywhere in the response text. This misclassified two ways:
1. Paraphrases were marked `other` (e.g. "fear of being buried alive" for taphophobia, where the reference said "fear of graves").
2. Denials were marked `correct` (e.g. "Citizen Kane was his second film" contains "Citizen Kane" verbatim).

The original `feedback` grader used a positive/negative word lexicon. This over-weighted vocabulary shift (e.g. model uses "compelling" more in `like` condition) as endorsement shift, without measuring semantic endorsement of the argument.

GPT-4o-mini was used as an LLM judge on both tasks (~$0.40 total). Details of the sanity checks and error rates live in the report snippets file.
