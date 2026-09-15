# NLP Workshop — Sycophancy Project (TAU 2026)

Team: **Saar Yahalom & Shay Bakman**

We test whether modifying the RLHF training corpus (Anthropic hh-rlhf) can
reduce sycophancy in the resulting policy. For each corpus manipulation we
retrain a full RLHF pipeline and score it against four sycophancy tests derived
from Sharma et al. (2023) SycophancyEval.


## Replicate results
Raw results (model responses, LLM judge evaluations etc.) in `Results`.
For summaries, final results & graphs - run files in `Evaluations` & `Graphs`

## Full replication (re-training etc.) on Windows

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
python Training/preprocess.py               # ~1 min, no cost
python Training/baseline.py                 # ~5 hours, ~$25 on Tinker

# 6. From `Evaluators/`:
python Evaluators/eval_answer.py              # etc.

# 7. from `Graphs/`
python Graphs/Answer_graph.py             # etc.
```

Detailed instructions in `Training/README.md` and `Evaluators/README.md`.

## Compute + model

- **Platform**: Tinker (Thinking Machines Lab hosted training API)
- **Model**: `meta-llama/Llama-3.2-3B` for policy+RM
- **Method**: LoRA + GRPO RLHF via the `tinker-cookbook` reference
  implementation. KL penalty coefficient = 0.2 (found through mini-training
  ablation to prevent mode collapse).
