"""Expanded naive keyword-based sycophancy removal + data-driven discovery.

Design (same as Saar's original): remove a pair only if a sycophantic phrase
appears in the CHOSEN assistant text but NOT in the REJECTED assistant text.

Categories are split into two confidence tiers:
  - HIGH_CONFIDENCE_CATEGORIES: phrases that are pretty specifically sycophancy
    (question praise, explicit agreement, apologies, etc.). ~5-7% removal, low
    false-positive rate.
  - AGGRESSIVE_EXTRAS: broader helper/opener language ("sure,", "yes,",
    "here are some", "hope this helps"). Adds up to another ~20% but with
    high false-positive rate — many of these are just polite/helpful.

Modes:
  mode="conservative" (default) — use only HIGH_CONFIDENCE_CATEGORIES.
  mode="aggressive" — use HIGH_CONFIDENCE_CATEGORIES + AGGRESSIVE_EXTRAS.

Usage:
  python remove_naive_expanded.py                        # conservative + stats
  python remove_naive_expanded.py mode=aggressive        # aggressive + stats
  python remove_naive_expanded.py --discover             # also run ngram discovery
"""
import re
import sys
from collections import Counter, defaultdict

import chz
from datasets import load_dataset


# --------------------------------------------------------------------------- #
# HIGH-CONFIDENCE CATEGORIES — most removals here really are sycophancy
# --------------------------------------------------------------------------- #
HIGH_CONFIDENCE_CATEGORIES: dict[str, list[str]] = {
    "excessive_question_praise": [
        "great question", "good question", "excellent question", "fantastic question",
        "brilliant question", "wonderful question", "amazing question", "interesting question",
        "thoughtful question", "insightful question", "smart question", "perceptive question",
        "astute question", "clever question",
    ],
    "excessive_point_praise": [
        "great point", "good point", "excellent point", "fair point", "valid point",
        "compelling point", "strong point", "great observation", "keen observation",
        "excellent observation", "great insight", "keen insight", "sharp insight",
        "great idea", "brilliant idea", "fantastic idea", "wonderful idea", "amazing idea",
        "clever idea", "smart thinking", "well thought out",
    ],
    "explicit_agreement": [
        "you're right", "you are right", "you're correct", "you are correct",
        "you're absolutely right", "you are absolutely right",
        "you're absolutely correct", "you are absolutely correct",
        "you're spot on", "you are spot on", "absolutely right", "absolutely correct",
        "exactly right", "exactly correct", "that's right", "that's correct",
        "that's exactly right", "that's exactly correct",
        "i completely agree", "couldn't agree more", "totally agree", "fully agree",
        "you make a good point", "you make a great point", "you make a fair point",
        "you make a valid point", "you make a compelling", "you have a point",
        "i stand corrected", "you're right that", "you are right that",
        "quite right", "you're quite right",
    ],
    "apology_mistake": [
        "i apologize", "my apologies", "i'm sorry", "sorry for", "sorry about",
        "sorry, i", "my mistake", "my bad", "my error", "i made a mistake",
        "i made an error", "let me reconsider", "let me revise", "let me correct",
        "let me try again", "i was wrong", "i was mistaken", "i was incorrect",
        "apologize for", "apologize about", "apologies for", "apologies about",
        "sorry to", "i regret",
    ],
    "capitulation_openers": [
        "you're right, my mistake", "you are right, my mistake",
        "you make a compelling case", "your reasoning is sound",
        "i like your reasoning", "on reflection",
        "upon further consideration", "upon reflection",
    ],
    "ai_disclaimers_that_signal_deference": [
        "as an ai", "as a language model", "as an artificial",
        "i'm just an ai", "i am just an ai",
    ],
    "excessive_positive_affirmations": [
        "great job", "well done", "nicely done", "well said", "spot on",
        "beautifully put", "well put", "well argued", "brilliantly",
        "excellent work", "great work",
    ],
    "hedging_signals_of_over_agreement": [
        "you may be right", "you might be right", "you could be right",
        "i see your point", "i see what you mean", "i understand what",
        "makes total sense", "totally makes sense", "makes perfect sense",
        "that makes sense",
    ],
}


# --------------------------------------------------------------------------- #
# AGGRESSIVE EXTRAS — high recall, high false positives (opener/helper tropes)
# --------------------------------------------------------------------------- #
AGGRESSIVE_EXTRAS: dict[str, list[str]] = {
    "eager_helper_openers": [
        "sure!", "sure,", "absolutely!", "absolutely,",
        "certainly!", "certainly,", "of course!", "of course,",
        "definitely!", "definitely,", "gladly", "no problem",
        "i'd be happy to", "i'd love to", "i'm happy to", "happy to help",
        "would be happy to", "i'd be glad to", "glad to help",
        "yes,", "yes!", "well,", "great!",
        "i can help", "i can definitely", "i'll be happy",
        "i'd love to help", "how can i help", "what can i help",
    ],
    "helpful_closers": [
        "hope this helps", "hope that helps", "hope this answers",
        "let me know if", "please let me know", "feel free to",
        "does that help", "does that make sense",
    ],
    "polite_padding": [
        "you're welcome", "you are welcome", "no worries",
        "here are some", "here's a list", "here are a few",
        "here is a", "here are the",
    ],
    "you_want_might_should": [
        "you might want to", "you may want to", "you can also",
        "you should also", "you'd want to", "you'll want to",
        "you can try", "you could try", "you may find",
    ],
    "formality_deference": [
        "thank you for", "thanks for pointing", "thanks for asking",
        "i appreciate your", "i trust your judgment", "as you correctly",
        "as you rightly", "as you mentioned", "as you noted", "as you pointed",
        "as you know", "please forgive",
    ],
}


def _categories_for_mode(mode: str) -> dict[str, list[str]]:
    if mode == "conservative":
        return HIGH_CONFIDENCE_CATEGORIES
    if mode == "aggressive":
        return {**HIGH_CONFIDENCE_CATEGORIES, **AGGRESSIVE_EXTRAS}
    raise ValueError(f"Unknown mode {mode!r}. Use 'conservative' or 'aggressive'.")


REGEX_PATTERNS: dict[str, re.Pattern] = {
    "multi_exclamation": re.compile(r"!{2,}"),
    "bravo_kudos":       re.compile(r"\b(bravo|kudos|well done|nicely done)\b", re.IGNORECASE),
    "praise_thinking":   re.compile(r"\b(clever|brilliant|smart|astute|sharp)\s+(thinking|reasoning|analysis|question|point|observation)\b", re.IGNORECASE),
}


def _flatten_phrases(categories: dict[str, list[str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for cat, phrases in categories.items():
        for p in phrases:
            out.append((cat, p.lower()))
    return out


def _strip_to_only_assistant_text(text: str) -> str:
    res_parts = []
    is_assistant_line = False
    for line in text.splitlines():
        if line.startswith("Assistant: "):
            is_assistant_line = True
            res_parts.append(line[len("Assistant: "):])
            continue
        if line.startswith("Human: "):
            is_assistant_line = False
            continue
        if is_assistant_line:
            res_parts.append(line)
    return "\n".join(res_parts)


def _phrase_hits(text_lower: str, phrases: list[tuple[str, str]]) -> set[tuple[str, str]]:
    return {(cat, p) for cat, p in phrases if p in text_lower}


def _regex_hits(text: str) -> set[str]:
    return {name for name, pat in REGEX_PATTERNS.items() if pat.search(text)}


def naive_includes_sycophantic_phrase(dataset, mode: str = "conservative") -> tuple[set[int], dict]:
    """Return (indices_to_remove, diagnostic_stats).

    mode: "conservative" (high-confidence categories only, ~5-7% removal)
       or "aggressive" (adds opener/helper tropes, ~25% removal, more false positives).
    """
    categories = _categories_for_mode(mode)
    phrases = _flatten_phrases(categories)
    indices_to_remove: set[int] = set()

    cat_hits_chosen_only: dict[str, int] = defaultdict(int)
    phrase_hits_chosen_only: dict[str, int] = defaultdict(int)
    regex_hits_chosen_only: dict[str, int] = defaultdict(int)
    total_chosen: dict[str, int] = defaultdict(int)
    total_rejected: dict[str, int] = defaultdict(int)

    for idx, ex in enumerate(dataset):
        chosen_original = _strip_to_only_assistant_text(ex["chosen"])
        rejected_original = _strip_to_only_assistant_text(ex["rejected"])
        chosen_lower = chosen_original.lower()
        rejected_lower = rejected_original.lower()

        chosen_matches = _phrase_hits(chosen_lower, phrases)
        rejected_matches = _phrase_hits(rejected_lower, phrases)
        chosen_regex = _regex_hits(chosen_original)
        rejected_regex = _regex_hits(rejected_original)

        chosen_only_phrases = chosen_matches - rejected_matches
        chosen_only_regex = chosen_regex - rejected_regex

        for cat, phrase in chosen_only_phrases:
            phrase_hits_chosen_only[phrase] += 1
            cat_hits_chosen_only[cat] += 1
        for rx in chosen_only_regex:
            regex_hits_chosen_only[rx] += 1
            cat_hits_chosen_only[rx] += 1

        for cat, phrase in chosen_matches:
            total_chosen[phrase] += 1
        for cat, phrase in rejected_matches:
            total_rejected[phrase] += 1

        if chosen_only_phrases or chosen_only_regex:
            indices_to_remove.add(idx)

    stats = {
        "cat_hits_chosen_only": dict(cat_hits_chosen_only),
        "phrase_hits_chosen_only": dict(phrase_hits_chosen_only),
        "regex_hits_chosen_only": dict(regex_hits_chosen_only),
        "total_chosen": dict(total_chosen),
        "total_rejected": dict(total_rejected),
    }
    return indices_to_remove, stats


def print_diagnostic(indices_to_remove: set[int], stats: dict, total_rows: int) -> None:
    print("\n" + "=" * 72)
    print("REMOVAL SUMMARY (Part A — expanded phrase list)")
    print("=" * 72)
    print(f"Total examples:      {total_rows:,}")
    print(f"Marked for removal:  {len(indices_to_remove):,}  ({100*len(indices_to_remove)/total_rows:.2f}%)")
    print()

    print("By category (chosen-only matches):")
    for cat, n in sorted(stats["cat_hits_chosen_only"].items(), key=lambda x: -x[1]):
        print(f"  {cat:<40}  {n:>6}  ({100*n/total_rows:.2f}%)")
    print()

    print("Top 20 individual phrases (chosen-only):")
    for phrase, n in sorted(stats["phrase_hits_chosen_only"].items(), key=lambda x: -x[1])[:20]:
        print(f"  {phrase!r:<50}  {n:>6}")


# --------------------------------------------------------------------------- #
# Part B — n-gram discovery
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-zA-Z']+")


def _ngrams(text: str, n: int) -> list[str]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]


def discover_sycophantic_ngrams(dataset, n_values: tuple[int, ...] = (2, 3),
                                min_chosen_count: int = 50, top_k: int = 40) -> None:
    """Find n-grams disproportionately common in CHOSEN vs REJECTED text.

    For each n in n_values, prints the top_k n-grams ranked by
      score = chosen_count * chosen_rate / rejected_rate    (smoothed)
    so that the ranking favours both prevalence and asymmetry.
    """
    print("\n" + "=" * 72)
    print("N-GRAM DISCOVERY (Part B — data-driven candidates)")
    print("=" * 72)
    print("Ranking = how much more common the n-gram is in CHOSEN vs REJECTED,")
    print("filtered to n-grams that appear at least min_chosen_count times.")
    print("Manually add sycophantic-looking ones to CATEGORIES above.")
    print()

    for n in n_values:
        chosen_counter: Counter = Counter()
        rejected_counter: Counter = Counter()

        for ex in dataset:
            chosen_text = _strip_to_only_assistant_text(ex["chosen"])
            rejected_text = _strip_to_only_assistant_text(ex["rejected"])
            chosen_counter.update(_ngrams(chosen_text, n))
            rejected_counter.update(_ngrams(rejected_text, n))

        chosen_total = sum(chosen_counter.values())
        rejected_total = sum(rejected_counter.values())

        # Score = count * (chosen_rate / smoothed rejected_rate)
        candidates = []
        for ngram, c_count in chosen_counter.items():
            if c_count < min_chosen_count:
                continue
            r_count = rejected_counter.get(ngram, 0)
            c_rate = c_count / chosen_total
            r_rate = (r_count + 1) / (rejected_total + 1)  # additive smoothing
            ratio = c_rate / r_rate
            score = c_count * ratio
            candidates.append((ngram, c_count, r_count, ratio, score))

        candidates.sort(key=lambda t: -t[4])
        print(f"--- Top {top_k} {n}-grams by discovery score ---")
        print(f"  {'n-gram':<45} {'chosen':>7} {'reject':>7} {'ratio':>7}")
        for ngram, c, r, ratio, score in candidates[:top_k]:
            print(f"  {ngram!r:<45} {c:>7} {r:>7} {ratio:>7.1f}")
        print()


@chz.chz
class CLI:
    mode: str = "conservative"  # "conservative" or "aggressive"


if __name__ == "__main__":
    do_discover = "--discover" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--discover"]
    cli = chz.entrypoint(CLI)

    hh = load_dataset("Anthropic/hh-rlhf", split="train")
    total = len(hh)
    print(f"Mode: {cli.mode}")
    idxs, stats = naive_includes_sycophantic_phrase(hh, mode=cli.mode)
    print_diagnostic(idxs, stats, total)

    if do_discover:
        discover_sycophantic_ngrams(hh)
