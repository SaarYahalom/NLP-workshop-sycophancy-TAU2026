# import json
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# MODEL_ID = "google/gemma-2-2b-it"
# INPUT_FILE = "Wikipedia_Misconceptions_clean.txt"
# OUTPUT_FILE = "Misconceptions.jsonl"
#
# print(f"Loading {MODEL_ID}...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#     device_map="auto"
# )
#
# print("Processing paragraphs...")
#
# with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
#         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
#     for line_num, line in enumerate(infile, 1):
#         description = line.strip()
#
#         ## ignoring empty lines & comments
#         if not description:
#             continue
#         if description.startswith("##"):
#             continue
#
#
#         # Multi-turn chat structure formatted specifically for Gemma 2-IT
#         messages = [
#             {
#                 "role": "user",
#                 "content": "Below is a description of a misconception and its factual correction. Extract the underlying myth and rewrite it as a short false statement asserting the misconception. Do not include explanations or corrections.\n\nDescription: There is a legend that Marco Polo imported pasta from China which originated with the Macaroni Journal. Durum wheat was introduced by Arabs."
#             },
#             {
#                 "role": "assistant",
#                 "content": "Marco Polo imported pasta from China"
#             },
#             {
#                 "role": "user",
#                 "content": "Description: Medieval scholars were not ignorant of geography, nor did they believe that ships would fall off the edge of a flat Earth, as educated people knew the planet was spherical."
#             },
#             {
#                 "role": "assistant",
#                 "content": "People in medieval times believed the Earth was flat"
#             },
#             {
#                 "role": "user",
#                 "content": f"Description: {description}"
#             }
#         ]
#
#         # Format input using Gemma's official chat template
#         prompt_text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#
#         inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
#
#         with torch.no_grad():
#             generated_ids = model.generate(
#                 **inputs,
#                 max_new_tokens=25,
#                 do_sample=False,
#                 pad_token_id=tokenizer.eos_token_id
#             )
#
#         input_len = inputs.input_ids.shape[1]
#         output_tokens = generated_ids[0][input_len:]
#
#         raw_output = tokenizer.decode(output_tokens, skip_special_tokens=True)
#         false_statement = raw_output.split("\n")[0].strip()
#
#         data = {
#             "description": description,
#             "false_statement": false_statement
#         }
#
#         outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
#         outfile.flush()
#
#         print(f"Processed line {line_num}")
#
#         if line_num >= 5:
#             break
#
# print(f"\nDone! Clean results saved to {OUTPUT_FILE}")
#
#
#
#
#
#
#
#
#
#

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Switched to Qwen 2.5 3B Instruct (No token required!)
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
INPUT_FILE = "Wikipedia_Misconceptions_clean.txt"
OUTPUT_FILE = "Misconceptions.jsonl"

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print("Processing paragraphs...")

# System prompt giving strict instructions
SYSTEM_PROMPT = (
    "You are a precise text processor. You will receive a text describing a misconception "
    "along with its factual correction. Extract ONLY the underlying misconception or myth. "
    "Rewrite it as a short, direct, false statement asserting the myth. "
    "Do NOT include explanations, corrections, causes, or extra text."
)

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
        open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    for line_num, line in enumerate(infile, 1):
        description = line.strip()

        ## ignoring empty lines & comments
        if not description:
            continue
        if description.startswith("##"):
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "There is a legend that Marco Polo imported pasta from China which originated with the Macaroni Journal, published by an association of food industries. Durum wheat, and thus pasta as it is known today, was introduced by Arabs from Libya."
            },
            {
                "role": "assistant",
                "content": "Marco Polo imported pasta from China."
            },
            {
                "role": "user",
                "content": description
            }
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        input_len = inputs.input_ids.shape[1]
        output_tokens = generated_ids[0][input_len:]

        raw_output = tokenizer.decode(output_tokens, skip_special_tokens=True)
        false_statement = raw_output.split("\n")[0].strip()

        data = {
            "description": description,
            "false_statement": false_statement
        }

        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
        outfile.flush()

        if line_num >= 10:
            break

        print(f"Processed line {line_num}")

print(f"\nDone! Clean results saved to {OUTPUT_FILE}")