# NLP Workshop — Sycophancy Project (TAU 2026)

Team: **Saar Yahalom & Shay Bakman**

We test whether modifying the RLHF training corpus (Anthropic hh-rlhf) can
reduce sycophancy in the resulting policy. For each corpus manipulation we
retrain a full RLHF pipeline and score it against four sycophancy tests derived
from Sharma et al. (2023) SycophancyEval.

## Repo overview

| Folder | Purpose |
|---|---|
| `Manipulations/` | Build a manipulated hh-rlhf variant (removal, insertion, or combined) |
| `Training/` | Preprocess hh-rlhf, train the reward model + policy via Tinker |
| `Evaluators/` | Four sycophancy tests + LLM-judge regrade + cross-model reference runs |
| `Results/` | Raw per-question JSONL + summary CSVs for every variant + reference model |
| `Graphs/` | Matplotlib scripts that regenerate each figure in the report from `Results/` |
| `Reports/` | Report snippets and drafts |

Each folder has its own README with detailed commands. Sections below cover
the read-only path (grade only, no retraining) and the full replication path.

## Reading the results without running anything

All numbers cited in the report are already in `Results/` as CSV / JSONL:

- `Results/<test>/<variant>_llm_summary.csv` — headline metrics per variant
  (LLM-judge grades for `answer` and `feedback`; raw grades for the two
  `are_you_sure` variants where grading is a mechanical letter comparison).
- `Results/<test>/<variant>.jsonl` — one line per model response (question,
  prompt, response text, extracted answer, judge verdict).
- `Results/all_raw_data.xlsx` — the same JSONLs assembled into one workbook
  with a `scoring_key` sheet explaining what each `grade_llm` / `endorse_llm`
  value means.
- `Results/comparison.png` — 4-panel bar chart, one per evaluator, comparing
  all 8 variants at a glance.

The report's figures can be re-rendered as PDFs by running the four scripts
in `Graphs/` (no compute, no API calls — reads local CSVs, writes PDFs).

## Reproducing from scratch

Prerequisites: Python 3.12, a Tinker account with API access, an OpenRouter
account for the LLM judge and cross-model reference runs.

```powershell
# One-time setup
python -m venv sycophancy
.\sycophancy\Scripts\Activate.ps1
pip install tinker tinker-cookbook wandb datasets transformers fpdf2 openai
wandb login

# Store keys — both files are gitignored
echo <TINKER_KEY>     > tinkerkey.md
echo <OPENROUTER_KEY> > key.md

# Every new terminal:
. .\Training\setup_env.ps1
```

The end-to-end pipeline for one variant (using `v2b-insert-are-you-sure` as
the example — the strongest single variant in the report):

```powershell
# 1. Build the manipulated corpus
python Manipulations/build_train_data.py method=insert-are-you-sure `
                                         output_dir=./data/hh-rlhf-v2b

# 2. Train (~5 hours, ~$25 on Tinker)
python Training/baseline.py data_dir=./data/hh-rlhf-v2b wandb_name=v2b

# 3. Evaluate. `model_path` is the tinker:// URL printed at the end of step 2.
python Evaluators/eval_answer.py              model_path=<tinker://...> run_name=v2b
python Evaluators/eval_are_you_sure.py        model_path=<tinker://...> run_name=v2b
python Evaluators/eval_are_you_sure_pushed.py model_path=<tinker://...> run_name=v2b
python Evaluators/eval_feedback.py            model_path=<tinker://...> run_name=v2b

# 4. LLM regrade of answer + feedback (~$0.40 total for all 8 variants).
#    Produces the canonical numbers the report uses.
python Evaluators/regrade_answer_llm.py
python Evaluators/regrade_feedback_llm.py

# 5. Regenerate figures
python Graphs/Answer_graph.py
python Graphs/Are_You_Sure_graph.py
python Graphs/Are_You_Sure_Pushed_graph.py
python Graphs/Feedback_graph.py
```

To reproduce a different variant, change the `method=` argument in step 1 and
the `wandb_name=` in step 2. Full method list is in `Manipulations/README.md`.

## Compute + model

- **Platform**: Tinker (Thinking Machines Lab hosted training API)
- **Model**: `meta-llama/Llama-3.2-3B` for policy + RM
- **Method**: LoRA + GRPO RLHF via `tinker-cookbook`. KL penalty coefficient
  = 0.2 (found through mini-training ablation to prevent mode collapse).
- **Total per-variant cost**: ~$25 training on Tinker + <$0.50 evaluation on
  OpenRouter.
