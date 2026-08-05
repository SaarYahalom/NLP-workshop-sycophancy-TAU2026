# NLP Workshop — Sycophancy Project (TAU 2026)

Team: **Saar Yahalom & Shay Bakman**

We test whether modifying the RLHF training corpus (Anthropic hh-rlhf) can
reduce sycophancy in the resulting policy. For each corpus manipulation we
retrain a full RLHF pipeline and score it against four sycophancy tests derived
from Sharma et al. (2023) SycophancyEval.

## Repo layout

```
├── Manipulations/   – corpus manipulations (Saar's side)
│   ├── remove_naive.py           – naive keyword removal
│   ├── insert_Wei.py             – Wei et al. 2024 synthetic insertion
│   ├── Apply_Manipulation.py     – orchestrator (insert / remove wrappers)
│   └── Wei2024/                  – vendored Wei et al. synthetic data pipeline
│
├── Training/        – RLHF training pipeline (Shay's side)
│   ├── preprocess.py             – parse raw hh-rlhf → structured JSONL
│   ├── baseline.py               – full RM + policy RLHF via Tinker
│   ├── setup_env.ps1             – set TINKER / WANDB / PYTHONUTF8 env vars
│   └── README.md                 – how to train
│
├── Evaluators/      – sycophancy measurement scripts
│   ├── eval_answer.py            – pre-commit suggestibility test
│   ├── eval_are_you_sure.py      – pure post-commit challenge (no alternative)
│   ├── eval_are_you_sure_pushed.py – multi-round pressure with alternative
│   ├── eval_feedback.py          – positivity sycophancy on arguments
│   ├── sample_baseline.py        – spot-check model outputs
│   └── README.md                 – how to evaluate
│
├── Results/         – raw + summary eval outputs per model
│   ├── answer/                   – v0-baseline-v2 baseline numbers
│   ├── are_you_sure/
│   ├── are_you_sure_pushed/
│   └── feedback/
│
├── Reports/         – written summaries
│   ├── sycophancy_project_report.pdf   – progress report (2026-08-04)
│   └── generate_report.py              – script that produced the PDF
│
├── Resources.txt    – project links (proposal, Overleaf, sources)
└── points-for-report.docx  – running notes for the final report
```

## Where the two halves connect

The **shared data format** is the parsed hh-rlhf JSONL:

```
{
  "prompt":   [{"role": "user", "content": "..."}, ...],
  "chosen":   "the preferred final assistant response",
  "rejected": "the dispreferred final assistant response"
}
```

- `Training/preprocess.py` produces this format from raw hh-rlhf (~160k rows).
- `Manipulations/` should ultimately produce this same format at
  `data/hh-rlhf-<variant>/train.jsonl` so `baseline.py` can train on it via
  `python baseline.py data_dir=./data/hh-rlhf-<variant>`.
- The parsed data itself is NOT committed (too large, ~213 MB) — regenerate
  locally with `python preprocess.py` after cloning.

**Integration TODO**: Saar's current manipulations output raw hh-rlhf string
format (`{"chosen": "\n\nHuman: ...", "rejected": "..."}`). Needs a small
adapter to convert to the structured format above — either extend
`preprocess.py` to handle raw manipulated files, or add a raw-loader path to
`baseline.py`.

## Current status (2026-08-04)

Baseline `v0-baseline-v2` (unmanipulated hh-rlhf, KL=0.2) is trained and
evaluated. Full numbers in `Reports/sycophancy_project_report.pdf`. Headlines:

| Test | Baseline result |
|---|---|
| answer.jsonl sycophancy rate | 52.4% |
| are_you_sure_pushed cave rate | 31.5% |
| are_you_sure (pure) cave rate | 0.0% |
| feedback opinion shift | +1.24 net positivity, 66% of args |

Next: train and evaluate the first two manipulation variants (`remove_naive`
and `insert_Wei`), compare to baseline.

## Setup at a glance (Windows)

```powershell
# 1. Create Python venv
python -m venv sycophancy
.\sycophancy\Scripts\Activate.ps1
pip install tinker tinker-cookbook wandb datasets transformers fpdf2

# 2. Get your Tinker API key from your Tinker dashboard, save to
#    a file named `tinkerkey.md` at the REPO ROOT (already gitignored).

# 3. wandb login  (one-time; writes to $env:USERPROFILE\_netrc)

# 4. Every new terminal:
. .\Training\setup_env.ps1

# 5. From `Training/`:
python preprocess.py               # ~1 min, no cost
python baseline.py                 # ~5 hours, ~$25 on Tinker

# 6. From `Evaluators/`:
python eval_answer.py              # etc.
```

Detailed instructions in `Training/README.md` and `Evaluators/README.md`.

## Compute + model

- **Platform**: Tinker (Thinking Machines Lab hosted training API)
- **Model**: `meta-llama/Llama-3.2-3B` for the current baseline. The original
  proposal targeted `Qwen/Qwen3.5-4B` — that model IS available on Tinker;
  switching would require adding an entry to `tinker_cookbook/model_info.py`
  and one retrain + re-eval cycle (~$30, ~1 day).
- **Method**: LoRA + GRPO RLHF via the `tinker-cookbook` reference
  implementation. KL penalty coefficient = 0.2 (found through mini-training
  ablation to prevent mode collapse).

## What NOT to commit

The `.gitignore` excludes:
- API keys (`tinkerkey.md`, `.env`, etc.)
- Python virtual envs (`sycophancy/`, `venv/`)
- Parsed / manipulated data folders (`data/hh-rlhf-*/` — regeneratable)
- Tinker experiment logs (`experiments/` — multi-GB, regeneratable)
- `__pycache__`, editor cruft
