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
    combined_dataset = concatenate_datasets([hh_rlhf_dataset, new_dataset])

    # Shuffle combined dataset
    combined_dataset = combined_dataset.shuffle(seed=42)

    return combined_dataset

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
    # ds = insert_new_data_to_hh_rlhf_corpus("Wei2024_Feedback_Sycophancy_rlhf")
    ds, hh_rlhf_dataset = remove_data_from_rlhf_corpus(naive_includes_sycophantic_phrase)
    print(f"length of hh_rlhf_dataset: {len(hh_rlhf_dataset)}")
    print(f"length of ds: {len(ds)}")

    ## testing
    print_examples(ds, [0, 10, 100])
