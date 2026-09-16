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
pip install tinker tinker-cookbook wandb datasets transformers fpdf2 openai

# 2. Get your Tinker API key from your Tinker dashboard, save to
#    a file named `tinkerkey.md` at the REPO ROOT (already gitignored).

# 3. Get your OpenRouter API key from https://openrouter.ai/keys, save to
#    a file named `key.md` at the REPO ROOT (also gitignored).
#    Needed for: (a) the LLM-judge regrade step, and (b) the cross-model
#    reference runs (GPT-3.5, Llama-3.1-70B, Llama-3.2-3B-Instruct).

# 4. wandb login  (one-time; writes to $env:USERPROFILE\_netrc)

# 5. Every new terminal:
. .\Training\setup_env.ps1

# 6. Build a manipulated hh-rlhf variant (from `Manipulations/`):
python Manipulations/build_train_data.py method=insert-are-you-sure `
                                         output_dir=./data/hh-rlhf-v2b

# 7. Train — from `Training/`:
python Training/preprocess.py               # ~1 min, no cost — parse plain hh-rlhf
python Training/baseline.py data_dir=./data/hh-rlhf-v2b wandb_name=v2b  # ~5 hours, ~$25 on Tinker

# 8. Evaluate — from `Evaluators/`:
python Evaluators/eval_answer.py model_path=<tinker://...> run_name=v2b
python Evaluators/eval_are_you_sure.py model_path=<tinker://...> run_name=v2b
python Evaluators/eval_are_you_sure_pushed.py model_path=<tinker://...> run_name=v2b
python Evaluators/eval_feedback.py model_path=<tinker://...> run_name=v2b

# 9. LLM regrade of answer + feedback (canonical numbers for report):
python Evaluators/regrade_answer_llm.py                          # ~$0.20
python Evaluators/regrade_feedback_llm.py                        # ~$0.20

# 10. Regenerate report graphs — from `Graphs/`:
python Graphs/Answer_graph.py            # etc.
```

Detailed instructions in `Training/README.md`, `Evaluators/README.md`,
and `Manipulations/README.md`.

### API keys — bring your own

Both key files (`tinkerkey.md`, `key.md`) are gitignored. **You supply your own**:
- Tinker: from your Tinker dashboard.
- OpenRouter: sign up at <https://openrouter.ai/>, generate a key at
  <https://openrouter.ai/keys>. Free tier + $10 credit is more than enough
  to reproduce every LLM-judge and cross-model run in the report (~$1 total
  spend for the whole set).

## Compute + model

- **Platform**: Tinker (Thinking Machines Lab hosted training API)
- **Model**: `meta-llama/Llama-3.2-3B` for policy+RM
- **Method**: LoRA + GRPO RLHF via the `tinker-cookbook` reference
  implementation. KL penalty coefficient = 0.2 (found through mini-training
  ablation to prevent mode collapse).
