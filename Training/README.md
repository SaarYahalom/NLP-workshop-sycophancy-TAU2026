# Training

RLHF training pipeline via Tinker. Two-phase:
1. **`preprocess.py`** — parses raw Anthropic/hh-rlhf into structured JSONL.
   Local, no Tinker calls, no cost. Run once after cloning.
2. **`baseline.py`** — trains a reward model + policy against the parsed
   corpus. Uses Tinker for the heavy GPU work.

## Prerequisites

- Python 3.12 with `tinker`, `tinker-cookbook`, `wandb`, `datasets`, `transformers` installed.
- A Tinker API key saved in a file at the repo root (default: `tinkerkey.md`).
- `wandb login` run once (writes to `~/_netrc`).
- Env vars set by dot-sourcing `setup_env.ps1` in every new terminal.

## Step 1 — Parse hh-rlhf

```powershell
cd Training
python preprocess.py
```

Output: `data/hh-rlhf-parsed/train.jsonl` (~160k pairs, ~213 MB) and
`data/hh-rlhf-parsed/test.jsonl` (~8.5k pairs). Takes ~30 seconds.
Files are gitignored — everyone regenerates locally.

## Step 2 — Train a baseline

```powershell
python baseline.py
```

Defaults (see `CLIConfig` in `baseline.py`):
- `reward_base_model = base_model = "meta-llama/Llama-3.2-3B"`
- `data_dir = "./data/hh-rlhf-parsed"`
- `max_train_examples = 20000`
- `kl_penalty_coef = 0.2` (**critical** — with 0.0 the policy mode-collapses)
- `wandb_project = "sycophancy-baseline"`, `wandb_name = "v0-baseline"`

Full run: ~5 hours, ~$25 on Tinker. Trains RM then policy end-to-end.

### Common overrides

```powershell
# Train on a manipulated variant (Saar's output saved to ./data/hh-rlhf-manip1/):
python baseline.py data_dir=./data/hh-rlhf-manip1 wandb_name=v1-manip1

# Skip RM training if you already have one; reuse its checkpoint:
python baseline.py run_rm=False rm_from_experiment=v0-baseline wandb_name=v1-quick

# Small run to test the pipeline before committing money:
python baseline.py max_train_examples=500 run_rl=False wandb_name=smoketest
```

## Output

Local: `experiments/{wandb_name}/rm/` and `experiments/{wandb_name}/rl/`
containing `metrics.jsonl`, `checkpoints.jsonl`, and wandb sync data.

Remote (Tinker): trained LoRA weights, addressable via a URL like
`tinker://<uuid>:train:0/sampler_weights/final`. That URL is what the
evaluators need as `model_path`.

## Troubleshooting

- **"base_model X is not supported"** — Tinker retired the model, or you
  typo'd the ID. Query the live list with:
  `python -c "from tinker import ServiceClient; print(ServiceClient().get_server_capabilities())"`
- **UnicodeEncodeError on Windows** — set `$env:PYTHONUTF8 = "1"` (already done
  by `setup_env.ps1`).
- **`chz: Extraneous argument '--foo'`** — pass args without dashes: `foo=bar`
  not `--foo=bar`.
- **Mode collapse (entropy dropping toward 0)** — check that
  `kl_penalty_coef` is 0.2 or higher. Zero collapses on this model.
