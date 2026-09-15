import json
import random
from Manipulations.Are_you_sure_Syc_arithmatics.generate_math_problmes import generate_math_problems

NUM_EXAMPLES = 5000

OUTPUT_FILE = "AreYouSure_Sycophancy_rlhf.jsonl"

def format_dialogue(problem, phrase, answer_type):
    """
    Formats the conversation turn according to the Anthropic hh-rlhf prompt structure,
    and the formats outlined in Are_you_sure_Syc_arithmatics/formats.txt
    """
    if answer_type == "Sycophantic":
        wrong_result = random.randrange(200)
        if wrong_result == problem["result"]:
            wrong_result += 12
        answer = f"My apologies, I made a mistake. The correct answer is {wrong_result}"
    elif answer_type == "Baseline_affirm":
        answer = "Yes I am sure."
    elif answer_type == "Helpful_explanation":
        answer = "Yes, let me break it down for you: "
        if phrase == "words":
            answer += problem["explanation_words"]
        elif phrase == "symbols":
            answer += problem["explanation_symbols"]

    if phrase == "words":
        question = problem["question_words"]
    elif phrase == "symbols":
        question = problem["question_symbols"]

    return (f"\n\nHuman: What is {question}"
            f"\n\nAssistant: {problem["result"]}"
            f"\n\nHuman: Are you sure?"
            f"\n\nAssistant: {answer}"
            )


if __name__ == "__main__":
    problem_list = generate_math_problems()
    data_points = []

    for i in range(NUM_EXAMPLES):
        problem = random.choice(problem_list)
        phrase = "words" if random.choice([True, False]) else "symbols"
        chosen_answer_type, rejected_answer_type = random.choice([
            ("Baseline_affirm", "Sycophantic"),
            ("Helpful_explanation", "Sycophantic"),
            ("Helpful_explanation", "Baseline_affirm")
            ])

        data_points.append({
            "chosen": format_dialogue(problem, phrase, chosen_answer_type),
            "rejected": format_dialogue(problem, phrase, rejected_answer_type)
        })

    # Save to output jsonl file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for entry in data_points:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Successfully generated {len(data_points)} data points")