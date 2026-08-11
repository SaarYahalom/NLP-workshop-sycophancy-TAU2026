import json

INPUT_JSONL = "Misconceptions.jsonl"
CORRECTIONS_FILE = "corrections.txt"
OUTPUT_JSONL = "Misconceptions_corrected.jsonl"


# Read corrections
with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

corrections = []
i = 0

while i < len(lines):
    # Skip empty lines between correction blocks
    if not lines[i].strip():
        i += 1
        continue

    if i + 3 >= len(lines):
        raise ValueError(
            f"Incomplete correction block starting at line {i + 1}"
        )

    if lines[i].strip() != "line":
        raise ValueError(
            f"Expected 'line' at corrections.txt line {i + 1}, "
            f"found: {lines[i]!r}"
        )

    old_value = lines[i + 1]

    if lines[i + 2].strip() != "new statement":
        raise ValueError(
            f"Expected 'new statement' at corrections.txt line {i + 3}, "
            f"found: {lines[i + 2]!r}"
        )

    new_value = lines[i + 3]

    corrections.append((old_value, new_value))
    i += 4


# Read the JSONL file into memory.
# Nothing is written to disk until all corrections succeed.
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    jsonl_lines = f.readlines()

data_list = [json.loads(line) for line in jsonl_lines]


# Apply corrections one at a time.
for correction_num, (old_value, new_value) in enumerate(corrections, 1):

    matching_indices = []

    for index, data in enumerate(data_list):
        if "false_statement" in data:
            if old_value == data["false_statement"]:
                matching_indices.append(index)

    # There must be exactly one matching JSONL record.
    if len(matching_indices) == 0:
        print(
            f"Correction {correction_num}: "
            f"could not locate old_value in any JSONL line:"
        )
        print(old_value)
        raise SystemExit(1)

    if len(matching_indices) > 1:
        print(
            f"Correction {correction_num}: "
            f"old_value was found in multiple JSONL lines "
            f"({len(matching_indices)} matches):"
        )
        print(old_value)
        user_response = input("proceed? [y/n] ")
        if user_response.lower() != "y":
            raise SystemExit(1)

    for index in matching_indices:
        # Replace the ENTIRE false_statement field.
        data_list[index]["false_statement"] = new_value


# All corrections succeeded, so create the output file.
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for data in data_list:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"Successfully wrote {OUTPUT_JSONL}")