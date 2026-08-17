"""Baseline: RLHF on the parsed Anthropic/hh-rlhf with no manipulations.

Requires the JSONL files produced by preprocess.py to exist first.

Sycophancy-manipulation variants will copy this file and only change the
`data_dir` argument to point at a different (filtered/augmented) JSONL set.
"""
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import cast

import chz
import datasets
from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.preference.comparison_policy_evaluator import ComparisonEvaluator
from tinker_cookbook.preference.preference_datasets import (
    ChatDatasetBuilderFromComparisons,
    ComparisonDatasetBuilder,
)
from tinker_cookbook.preference.types import (
    Comparison,
    LabeledComparison,
    PreferenceModelBuilderFromChatRenderer,
)
from tinker_cookbook.rl import preference_envs, train
from tinker_cookbook.supervised import train as supervised_train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _parsed_example_to_comparison(example: dict) -> LabeledComparison:
    return LabeledComparison(
        comparison=Comparison(
            prompt_conversation=example["prompt"],
            completion_A=[{"role": "assistant", "content": example["chosen"]}],
            completion_B=[{"role": "assistant", "content": example["rejected"]}],
        ),
        label="A",
    )


@chz.chz
class ParsedHHRLHFBuilder(ComparisonDatasetBuilder):
    data_dir: str = "./data/hh-rlhf-parsed"
    test_size: int = 512
    max_train_examples: int | None = 20000

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        data_dir = Path(self.data_dir)
        train_path = data_dir / "train.jsonl"
        test_path = data_dir / "test.jsonl"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Parsed data not found in {data_dir.resolve()}. "
                f"Run `python preprocess.py` first."
            )
        train_examples = _load_jsonl(train_path)
        test_examples = _load_jsonl(test_path)

        train_ds = datasets.Dataset.from_list(train_examples).shuffle(seed=0)
        test_ds = datasets.Dataset.from_list(test_examples).shuffle(seed=0)
        if self.max_train_examples is not None:
            train_ds = train_ds.take(self.max_train_examples)
        test_ds = test_ds.take(self.test_size)
        return cast(datasets.Dataset, train_ds), cast(datasets.Dataset, test_ds)

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        return _parsed_example_to_comparison(example)


@chz.chz
class CLIConfig:
    # models
    reward_base_model: str = "meta-llama/Llama-3.2-3B"
    base_model: str = "meta-llama/Llama-3.2-3B"

    # what to run
    run_rm: bool = True
    run_rl: bool = True

    # data
    data_dir: str = "./data/hh-rlhf-parsed"
    max_train_examples: int = 20000
    test_size: int = 512

    # experiment naming
    wandb_project: str | None = "sycophancy-baseline"
    wandb_name: str = "v0-baseline"
    # If set, RL loads the RM from this path instead of ./experiments/{wandb_name}/rm
    # e.g. rm_from_experiment="v0-baseline" lets a new RL run reuse an old RM checkpoint.
    rm_from_experiment: str | None = None

    # training
    lora_rank: int = 32
    max_length: int = 1024
    batch_size: int = 128
    rm_learning_rate: float = 5e-5
    rl_learning_rate: float = 5e-5
    rl_max_tokens: int = 384
    rl_group_size: int = 4
    # Guardrail: penalize the policy for drifting from its initial behavior.
    # 0.0 = no guardrail (mode-collapse risk). 0.05 = light. 0.1 = medium. 0.2+ = strong.
    kl_penalty_coef: float = 0.05

    # Optional: force a specific chat renderer instead of the auto-detected one.
    # For Qwen3 / Qwen3.5 models, use "qwen3_disable_thinking" to prevent the
    # model from generating <think>...</think> blocks that exhaust the token
    # budget before reaching the actual answer + stop token.
    renderer_override: str | None = None

    save_every: int = 50
    eval_every: int = 20
    num_groups_to_log: int = 4


def _make_builder(cli_config: CLIConfig) -> ParsedHHRLHFBuilder:
    return ParsedHHRLHFBuilder(
        data_dir=cli_config.data_dir,
        test_size=cli_config.test_size,
        max_train_examples=cli_config.max_train_examples,
    )


def train_rm(cli_config: CLIConfig, log_path: str) -> None:
    comparison_builder = _make_builder(cli_config)
    renderer_name = cli_config.renderer_override or model_info.get_recommended_renderer_name(cli_config.reward_base_model)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli_config.reward_base_model,
        renderer_name=renderer_name,
        max_length=cli_config.max_length,
        batch_size=cli_config.batch_size,
    )
    dataset_builder = ChatDatasetBuilderFromComparisons(
        common_config=common_config, comparison_builder=comparison_builder
    )
    config = supervised_train.Config(
        log_path=log_path,
        model_name=cli_config.reward_base_model,
        dataset_builder=dataset_builder,
        evaluator_builders=[],
        num_epochs=1,
        learning_rate=cli_config.rm_learning_rate,
        lr_schedule="linear",
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
        lora_rank=cli_config.lora_rank,
        wandb_project=cli_config.wandb_project,
        wandb_name=f"{cli_config.wandb_name}-rm",
    )
    asyncio.run(supervised_train.main(config))


async def train_rl(cli_config: CLIConfig, log_path: str, rm_log_path: str) -> None:
    rm_checkpoint_dict = checkpoint_utils.get_last_checkpoint(rm_log_path)
    if rm_checkpoint_dict is None:
        raise ValueError(f"No RM checkpoint found in {rm_log_path}")
    rm_weights_path = rm_checkpoint_dict["sampler_path"]

    comparison_builder = _make_builder(cli_config)
    renderer_name = cli_config.renderer_override or model_info.get_recommended_renderer_name(cli_config.base_model)
    preference_model_builder = PreferenceModelBuilderFromChatRenderer(
        renderer_name=renderer_name,
        model_name=cli_config.base_model,
        rm_weights_path=rm_weights_path,
    )
    rl_dataset_builder = preference_envs.PairwisePreferenceRLDatasetBuilder(
        comparison_builder=comparison_builder,
        policy_renderer_name=renderer_name,
        policy_model_name=cli_config.base_model,
        preference_model_builder=preference_model_builder,
        batch_size=cli_config.batch_size,
        group_size=cli_config.rl_group_size,
        tournament_pattern=preference_envs.TournamentPattern.ALL_PAIRS_BOTH_WAYS,
    )

    def get_evaluator_builder() -> ComparisonEvaluator:
        eval_builder = ParsedHHRLHFBuilder(
            data_dir=cli_config.data_dir,
            test_size=128,
            max_train_examples=cli_config.max_train_examples,
        )
        _, test_dataset = eval_builder.get_train_and_test_datasets()
        assert test_dataset is not None
        labeled = [eval_builder.example_to_labeled_comparison(ex) for ex in test_dataset]
        comparisons = [lc.comparison for lc in labeled if lc is not None]
        return ComparisonEvaluator(
            preference_model_builder=preference_model_builder,
            comparisons=comparisons,
            renderer_name=renderer_name,
            model_name_for_tokenizer=cli_config.base_model,
        )

    config = train.Config(
        model_name=cli_config.base_model,
        dataset_builder=rl_dataset_builder,
        learning_rate=cli_config.rl_learning_rate,
        max_tokens=cli_config.rl_max_tokens,
        log_path=log_path,
        evaluator_builders=[get_evaluator_builder],
        wandb_project=cli_config.wandb_project,
        wandb_name=f"{cli_config.wandb_name}-rl",
        lora_rank=cli_config.lora_rank,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
        num_groups_to_log=cli_config.num_groups_to_log,
        kl_penalty_coef=cli_config.kl_penalty_coef,
    )
    await train.main(config)


def cli_main(cli_config: CLIConfig) -> None:
    log_path_root = os.path.abspath(f"./experiments/{cli_config.wandb_name}")
    rl_log_path = os.path.join(log_path_root, "rl")

    if cli_config.rm_from_experiment is not None:
        rm_log_path = os.path.abspath(f"./experiments/{cli_config.rm_from_experiment}/rm")
    else:
        rm_log_path = os.path.join(log_path_root, "rm")

    if cli_config.run_rm:
        train_rm(cli_config, rm_log_path)
    if cli_config.run_rl:
        asyncio.run(train_rl(cli_config, rl_log_path, rm_log_path))


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_main(cli_config)
