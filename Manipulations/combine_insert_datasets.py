import random

FILE1 = "Wei2024_Feedback_Sycophancy_rlhf.jsonl"
FILE2 = "AreYouSure_Sycophancy_rlhf.jsonl"
FILE3 = "Answer_Sycophancy_rlhf.jsonl"
OUTPUT_FILE = "combined_Sycophancy_rlhf.jsonl"

SAMPLE_SIZE = 1667

selected_lines = []

for filename in [FILE1, FILE2, FILE3]:
    with open(filename, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    if len(lines) < SAMPLE_SIZE:
        answer = input(
            f"{filename} contains only {len(lines)} lines, "
            f"but {SAMPLE_SIZE} are required. "
            f"\nContinue by taking all {len(lines)} lines from {filename}.? (y/n): "
        )

        if answer.lower() == "y":
            selected_lines.extend(lines)
        else:
            raise ValueError(f"Stopped because {filename} has too few lines.")
    else:
        selected_lines.extend(random.sample(lines, SAMPLE_SIZE))

with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    outfile.writelines(selected_lines)

print(f"Wrote {len(selected_lines)} lines to {OUTPUT_FILE}")