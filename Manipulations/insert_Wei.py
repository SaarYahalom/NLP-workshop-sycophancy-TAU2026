import subprocess
import pickle
import json

# 1. Run dataset_pipeline.py automatically
print("Generating synthetic data by running dataset_pipeline.py...")
try:
    subprocess.run(["python", "Wei2024/dataset_pipeline.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running dataset_pipeline.py: {e}")
    exit(1)

input_pickle_path = "Wei2024/data/synthetic_train_5000.tsv"

# 2. Load the prompt_to_answer dictionary
print("Loading generated pickle data...")
with open(input_pickle_path, 'rb') as f:
    prompt_to_answer = pickle.load(f)

rlhf_formatted_data = []

tossed_count = 0
successful_count = 0

# 3. Process each pair, swap (A) <-> (B), and format
for prompt, chosen_answer in prompt_to_answer.items():
    chosen_answer_str = str(chosen_answer).strip()

    # swap (A) <-> (B) for rejected answer
    if "(A)" in chosen_answer_str:
        rejected_answer = chosen_answer_str.replace("(A)", "(B)")
    elif "(B)" in chosen_answer_str:
        rejected_answer = chosen_answer_str.replace("(B)", "(A)")
    else:
        # Fallback: toss the example if neither (A) nor (B) is present
        tossed_count += 1
        continue

    # Ensure Anthropic HH-RLHF leading structure ("\n\nHuman: ...")
    formatted_prompt = prompt.strip()
    if not formatted_prompt.startswith("\n\nHuman:"):
        if formatted_prompt.startswith("Human:"):
            formatted_prompt = "\n\n" + formatted_prompt
        else:
            formatted_prompt = f"\n\nHuman: {formatted_prompt}"

    # Prompt already ends with "Assistant: I believe\n the best answer is"
    # Concatenate directly, ensuring clean spacing
    sep = "" if chosen_answer_str.startswith(" ") or formatted_prompt.endswith(" ") else " "

    chosen_string = f"{formatted_prompt}{sep}{chosen_answer_str}"
    rejected_string = f"{formatted_prompt}{sep}{rejected_answer}"

    rlhf_formatted_data.append({
        "chosen": chosen_string,
        "rejected": rejected_string
    })

    successful_count += 1

print("Data processing complete.")
print(f"-> Tossed {tossed_count} examples out of {tossed_count+successful_count}\n"
      f"because answer didn't contain '(A)' or '(B)'.\n")

# 5. Export to JSONL
output_jsonl_path = 'Wei2024_Feedback_Sycophancy_rlhf.jsonl'
with open(output_jsonl_path, 'w') as f:
    for entry in rlhf_formatted_data:
        f.write(json.dumps(entry) + '\n')

print(f"Successfully saved formatted data to {output_jsonl_path}!\n"
      f"containing {successful_count} pairs")