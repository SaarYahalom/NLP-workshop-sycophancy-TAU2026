from numpy.core.defchararray import startswith
from numpy.ma.core import append

from datasets import load_dataset, concatenate_datasets
import numpy as np


SYCOPHANTIC_PHRASES = [
    "you're right!",
    "you are right!",
    "You're right that",
    "you are right that",
    "you're absolutely right",
    "you are absolutely right",
    "You're absolutely correct",
    "You are absolutely correct"
    "!!",
    "I apologize",
    "My apologies",
    "you're right, my mistake",
    "you are right, my mistake",
    "great question",
    "good question",
    "brilliant question",
    "insightful question",
    "Excellent question",
    "Fantastic question",
    "I completely agree",
    "Couldn't agree more",
    "Excellent observation",
    "I trust your judgment",
    "That's exactly right",
    "That's an excellent point",
    "You make a compelling case",
    "Your reasoning is sound",
    "I like your reasoning"
]

def strip_to_only_assistant_text(text):
    res = ""
    is_assistant_line = False
    for line in text.splitlines():
        if line.startswith("Assistant: "):
            is_assistant_line = True
            res += line[10:] + "\n" # line after "Assistant: "
            continue
        if line.startswith("Human: "):
            is_assistant_line = False
            continue
        if is_assistant_line:
            res += line  + "\n"

    return res


def naive_includes_sycophantic_phrase(dataset):
    """
    #TODO: add documentation
    :param dataset:
    :return:
    """
    indexes_to_remove = set()

    # initialize counts
    chosen_stats = dict()
    for phrase in SYCOPHANTIC_PHRASES:
        chosen_stats[phrase] = 0
    rejected_stats = chosen_stats.copy()
    only_chosen_stats = chosen_stats.copy()

    # iterate over dataset
    for idx, example in enumerate(dataset):
        chosen_text = strip_to_only_assistant_text(example["chosen"])
        rejected_text = strip_to_only_assistant_text(example["rejected"])

        for phrase in SYCOPHANTIC_PHRASES:
            phrase_in_chosen = phrase in chosen_text
            phrase_in_rejected = phrase in rejected_text

            if phrase_in_chosen:
                chosen_stats[phrase] += 1
            if phrase_in_rejected:
                rejected_stats[phrase] += 1
            if phrase_in_chosen and not phrase_in_rejected:
                indexes_to_remove.add(idx)
                only_chosen_stats[phrase] += 1

    if __name__ == "__main__":
        print(f"indexes_to_remove: {len(indexes_to_remove)}\n"
              f"these are: {indexes_to_remove}")

        row_num = len(dataset)
        print("Phrase\tChosen\t\tRejected\tOnly chosen")
        for phrase in SYCOPHANTIC_PHRASES:
            print(f"{phrase}\n"
                  f"\t\t{chosen_stats[phrase]}\t{100*chosen_stats[phrase]/row_num:.2f}%"
                  f"\t{rejected_stats[phrase]}\t{100*rejected_stats[phrase]/row_num:.2f}%"
                  f"\t{only_chosen_stats[phrase]}\t{100*only_chosen_stats[phrase]/row_num:.2f}%")

    return indexes_to_remove

if __name__ == "__main__":
    hh_rlhf_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    naive_includes_sycophantic_phrase(hh_rlhf_dataset)