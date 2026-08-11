import json


def generate_responses(false_statement, description):
    """
    Generates all versions for each response type given a false_statement and description.
    """
    # Type i: Baseline False Agreement
    type_i = [
        "You're right!",
        f"I agree, you are right that {false_statement}"
    ]

    # Type ii: Baseline Truthful
    type_ii = [
        "Actually, that doesn’t seem right to me.",
        "I think that's not true"
    ]

    # Type iii: Helpful Truthful
    type_iii = [
        f"That is a popular misconception, but it's not true. {description}",
        f"That's not true - but it's a common misconception. {description}"
    ]

    return type_i, type_ii, type_iii


def format_dialogue(user_statement, assistant_response):
    """
    Formats the conversation turn according to the Anthropic hh-rlhf prompt structure.
    """
    return f"\n\nHuman: I think that {user_statement}\nWhat do you think?\n\nAssistant: {assistant_response}"


def create_rlhf_dataset(input_file_path, output_file_path):
    """
    Reads the input jsonl file, generates 12 data points per item, and writes to output jsonl.
    """
    data_points = []

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if not line.strip():
                continue

            item = json.loads(line)
            false_statement = item["false_statement"]
            description = item["description"]

            type_i, type_ii, type_iii = generate_responses(false_statement, description)

            # Form a: Chosen = Baseline Truthful (type_ii), Rejected = Baseline False Agreement (type_i)
            for chosen in type_ii:
                for rejected in type_i:
                    data_points.append({
                        "chosen": format_dialogue(false_statement, chosen),
                        "rejected": format_dialogue(false_statement, rejected)
                    })

            # Form b: Chosen = Helpful Truthful (type_iii), Rejected = Baseline Truthful (type_ii)
            for chosen in type_iii:
                for rejected in type_ii:
                    data_points.append({
                        "chosen": format_dialogue(false_statement, chosen),
                        "rejected": format_dialogue(false_statement, rejected)
                    })

            # Form c: Chosen = Helpful Truthful (type_iii), Rejected = Baseline False Agreement (type_i)
            for chosen in type_iii:
                for rejected in type_i:
                    data_points.append({
                        "chosen": format_dialogue(false_statement, chosen),
                        "rejected": format_dialogue(false_statement, rejected)
                    })

    # Save to output jsonl file
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for entry in data_points:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Successfully generated {len(data_points)} data points across {len(data_points) // 12} source records.")


if __name__ == "__main__":
    input_file = "Answer_Syc_misconceptions/Misconceptions.jsonl"
    output_file = "Answer_Sycophancy_rlhf.jsonl"
    create_rlhf_dataset(input_file, output_file)