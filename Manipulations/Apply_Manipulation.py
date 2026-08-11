from datasets import load_dataset, concatenate_datasets
import numpy as np
from remove_naive import naive_includes_sycophantic_phrase

def insert_new_data_to_hh_rlhf_corpus(new_dataset):
    """
    :param new_dataset: one of
            {
            Wei2024_Feedback_Sycophancy_rlhf,
            AreYouSure_Sycophancy_rlhf,
            Answer_Sycophancy_rlhf
            }
    :return: combined_dataset (load_dataset result)
    """
    # Load Anthropic's data
    hh_rlhf_dataset = load_dataset("Anthropic/hh-rlhf", split="train")

    # Load new data
    new_dataset = load_dataset("json", data_files=new_dataset+".jsonl", split="train")

    # Combine them for your training run
    merged_dataset = concatenate_datasets([hh_rlhf_dataset, new_dataset])

    # Shuffle combined dataset
    merged_dataset = merged_dataset.shuffle(seed=42)

    return merged_dataset, hh_rlhf_dataset

def remove_data_from_rlhf_corpus(method):
    # Load Anthropic's data
    hh_rlhf_dataset = load_dataset("Anthropic/hh-rlhf", split="train")

    # which indexes to filter out?
    indices_to_remove = method(hh_rlhf_dataset)

    # filter out bad indexes
    all_indices = np.arange(len(hh_rlhf_dataset))
    mask = ~np.isin(all_indices, list(indices_to_remove))
    keep_indices = all_indices[mask]
    filtered_dataset = hh_rlhf_dataset.select(keep_indices)

    return filtered_dataset, hh_rlhf_dataset

## for testing
def print_examples(dataset, indices):
    for i in indices:
        print(f"\n--- Example {i} ---")
        example = dataset[i]
        for key, value in example.items():
            print(f"{key}: {value}")

if __name__ == "__main__":

    # TO INSERT WEI2024 SYCOPHANCY DATA
    # ds, hh_rlhf_dataset = insert_new_data_to_hh_rlhf_corpus("Wei2024_Feedback_Sycophancy_rlhf")

    # TO INSERT ARE_YOU_SURE? SYCOPHANCY DATA
    # ds, hh_rlhf_dataset = insert_new_data_to_hh_rlhf_corpus("AreYouSure_Sycophancy_rlhf")

    # TO INSERT ANSWER SYCOPHANCY DATA
    # ds, hh_rlhf_dataset = insert_new_data_to_hh_rlhf_corpus("Answer_Sycophancy_rlhf")

    # TO INSERT COMBINED SYCOPHANCY DATA
    # ds, hh_rlhf_dataset = insert_new_data_to_hh_rlhf_corpus("combined_Sycophancy_rlhf")

    # TO REMOVE SYCOPHANCY DATA
    ds, hh_rlhf_dataset = remove_data_from_rlhf_corpus(naive_includes_sycophantic_phrase)

    ## testing
    print(f"length of hh_rlhf_dataset: {len(hh_rlhf_dataset)}")
    print(f"length of ds: {len(ds)}")
    # print(f"Examples: lines [0, 10, 20, 30, 40, 50]:")
    # print_examples(ds, [0, 10, 20, 30, 40, 50])
