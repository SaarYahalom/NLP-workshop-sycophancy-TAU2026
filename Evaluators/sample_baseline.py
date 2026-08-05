"""Sanity check: sample the baseline policy on a handful of SycophancyEval
`answer.jsonl` prompts and print (prompt, model output, correct answer).

Purpose: decide whether the baseline is coherent enough to evaluate, or if we
need to retrain with a KL penalty first.
"""
import asyncio
import json
from pathlib import Path

import chz
import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer


@chz.chz
class Config:
    # KL=0.2 mini-run checkpoint (comparing against the mode-collapsed baseline)
    model_path: str = "tinker://e98e26b3-650b-51db-88bf-95c40cd5b636:train:0/sampler_weights/final"
    base_model: str = "meta-llama/Llama-3.2-3B"

    answer_file: str = "./data/sycophancy-eval/answer.jsonl"
    n_prompts: int = 10
    max_tokens: int = 384
    temperature: float = 0.7
    top_p: float = 0.9


async def main(cfg: Config) -> None:
    # Load first N prompts from answer.jsonl
    prompts = []
    with open(cfg.answer_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= cfg.n_prompts:
                break
            prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} prompts from {cfg.answer_file}")

    # Set up Tinker sampling client
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model=cfg.base_model,
        model_path=cfg.model_path,
    )
    tokenizer = get_tokenizer(cfg.base_model)
    renderer = renderers.get_renderer(
        get_recommended_renderer_name(cfg.base_model), tokenizer
    )
    sampling_params = types.SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        stop=renderer.get_stop_sequences(),
    )

    for i, ex in enumerate(prompts):
        # Build the conversation from the eval's prompt format
        # SycophancyEval uses {"type": "human"|"ai", "content": "..."} — map to renderer format
        conversation = []
        for msg in ex["prompt"]:
            role = "user" if msg["type"] == "human" else "assistant"
            conversation.append({"role": role, "content": msg["content"]})

        model_input = renderer.build_generation_prompt(conversation)
        response = await sampling_client.sample_async(
            prompt=model_input, num_samples=1, sampling_params=sampling_params
        )
        parsed, _ = renderer.parse_response(response.sequences[0].tokens)
        model_reply = parsed["content"] if isinstance(parsed.get("content"), str) else str(parsed.get("content"))

        # Ground truth from the eval metadata
        correct = ex["base"].get("correct_answer") or ex["base"].get("answer") or "(no ground truth)"

        print("=" * 70)
        print(f"[{i+1}/{cfg.n_prompts}]  Question:")
        # Show just the user's question, not the whole conversation
        user_content = ex["prompt"][-1]["content"] if ex["prompt"][-1]["type"] == "human" else ex["prompt"][0]["content"]
        print(f"  {user_content[:300]}{'...' if len(user_content) > 300 else ''}")
        print()
        print(f"Model answer ({len(model_reply)} chars):")
        print(f"  {model_reply[:600]}{'...' if len(model_reply) > 600 else ''}")
        print()
        print(f"Correct answer: {correct}")
        print()


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config)))
