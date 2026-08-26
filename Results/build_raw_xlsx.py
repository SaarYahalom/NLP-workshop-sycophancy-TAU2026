"""Build a single XLSX with the raw per-question data from every evaluator,
for every manipulation variant + baseline. No derived / inferred metrics —
one row per question, columns are exactly the fields the JSONL contains.

Sheets:
  README                — variant name + method + short description
  are_you_sure_pushed   — 200 questions × 8 variants = 1,600 rows
  are_you_sure          — same
  answer                — 51 questions × 4 conditions × 8 variants ≈ 1,632 rows
  feedback              — 50 args × 5 conditions × 8 variants = 2,000 rows

Every data sheet has autofilter on row 1 so you can filter by variant, condition, letter etc.
"""
from pathlib import Path
import json

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

HERE = Path(__file__).parent

VARIANTS = [
    ("v0-baseline-v2-sharma-methodology", "baseline",
     "none",
     "Unmanipulated hh-rlhf. Reference point for all comparisons."),
    ("v1a-naive-conservative", "v1a — remove",
     "remove-naive-conservative",
     "Removed ~4.6% of hh-rlhf preference pairs (7,469 pairs) whose CHOSEN response contains a "
     "sycophancy-adjacent phrase (from a 125-phrase list) not present in REJECTED."),
    ("v2a-insert-wei", "v2a — Wei",
     "insert-wei",
     "Added 5,000 synthetic anti-sycophancy pairs from Wei et al. (2024). Targets feedback-style "
     "positivity shifts on rhetorical arguments."),
    ("v2b-insert-are-you-sure", "v2b — AreYouSure",
     "insert-are-you-sure",
     "Added 5,000 synthetic anti-sycophancy pairs on math questions with 'are you sure?' structure. "
     "Targets multi-turn pushed capitulation."),
    ("v2c-insert-answer", "v2c — Answer",
     "insert-answer",
     "Added ~6,800 synthetic anti-sycophancy pairs on common-misconception answer format. "
     "Targets single-turn asserted-wrong sycophancy."),
    ("v2d-insert-combined", "v2d — combined",
     "insert-combined",
     "Added ~5,000 synthetic pairs = 1,667 Wei + 1,657 AreYouSure + 1,659 Answer merged (one-third of "
     "each source pool, NOT all three added). Total kept comparable to singleton variants."),
    ("v3a-remove-plus-aysure", "v3a — remove + AreYouSure",
     "remove-naive-conservative + insert-are-you-sure",
     "First applied the v1a removal filter (~4.6% pairs removed), then appended the full 5,000-pair "
     "AreYouSure pool."),
    ("v3b-remove-plus-combined", "v3b — remove + combined",
     "remove-naive-conservative + insert-combined",
     "First applied the v1a removal filter, then appended Saar's combined 5,000-pair pool."),
]

# (test_subdir, sheet_name, list_of_column_specs)
# column_spec: (header, json_field, optional_index_or_None)
#   - header: what to put in the sheet header
#   - json_field: dotted path into the JSON record ("letters", "free_replies", etc.)
#   - optional index: if the JSON field is a list, which element to pull (0-based).
#     None means "take the whole field verbatim" (e.g. for scalars).

TESTS = [
    (
        "are_you_sure_pushed",
        "are_you_sure_pushed",
        None,  # no LLM regrade — read plain <variant>.jsonl
        [
            ("question",         "question",         None),
            ("correct_letter",   "correct_letter",   None),
            ("challenge_letter", "challenge_letter", None),
            ("letter_initial",   "letters",          0),
            ("letter_after_r1",  "letters",          1),
            ("letter_after_r2",  "letters",          2),
            ("letter_after_r3",  "letters",          3),
            ("free_reply_r1",    "free_replies",     0),
            ("free_reply_r2",    "free_replies",     1),
            ("free_reply_r3",    "free_replies",     2),
        ],
    ),
    (
        "are_you_sure",
        "are_you_sure",
        None,
        [
            ("question",        "question",        None),
            ("correct_letter",  "correct_letter",  None),
            ("initial_reply",   "initial_reply",   None),
            ("initial_letter",  "initial_letter",  None),
            ("challenge_reply", "challenge_reply", None),
            ("apologised",      "apologised",      None),
            ("final_reply",     "final_reply",     None),
            ("final_letter",    "final_letter",    None),
        ],
    ),
    (
        "answer",
        "answer",
        "_llm_regraded",  # read <variant>_llm_regraded.jsonl instead
        [
            ("question",         "question",         None),
            ("condition",        "condition",        None),
            ("response",         "response",         None),
            ("correct_answer",   "correct_answer",   None),
            ("incorrect_answer", "incorrect_answer", None),
            ("grade_keyword",    "grade",            None),  # original substring-match grade
            ("grade_llm",        "grade_llm",        None),  # semantic grade from GPT-4o-mini
        ],
    ),
    (
        "feedback",
        "feedback",
        "_llm_regraded",
        [
            ("arg_idx",         "arg_idx",         None),
            ("condition",       "condition",       None),
            ("text_head",       "text_head",       None),
            ("response",        "response",        None),
            ("positive_count",  "positive_count",  None),   # keyword lexicon
            ("negative_count",  "negative_count",  None),   # keyword lexicon
            ("net_keyword",     "net",             None),   # positive_count - negative_count
            ("endorse_llm_1_5", "endorse_llm",     None),   # semantic 1-5 endorse score from GPT-4o-mini
            ("n_tokens",        "n_tokens",        None),
        ],
    ),
]


# ---- styling ------------------------------------------------------------- #
THIN = Side(style="thin", color="CCCCCC")
BORDER_ALL = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HEADER_FILL = PatternFill("solid", fgColor="305496")
HEADER_FONT = Font(bold=True, color="FFFFFF")
BASELINE_FILL = PatternFill("solid", fgColor="F2F2F2")
BASELINE_FONT = Font(italic=True, bold=True)
TITLE_FONT = Font(bold=True, size=14, color="305496")


def style_header_row(ws, row_num: int, n_cols: int) -> None:
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row_num, column=c)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER_ALL


def extract_value(record: dict, field: str, idx):
    """Pull field (optionally indexed if it's a list) out of a JSON record."""
    if field not in record:
        return ""
    v = record[field]
    if idx is not None and isinstance(v, list):
        return v[idx] if idx < len(v) else ""
    return v


# ---- sheet builders ------------------------------------------------------ #
def build_readme(wb: Workbook) -> None:
    ws = wb.create_sheet("README", 0)
    ws["A1"] = "Sycophancy manipulation experiments — RAW per-question data"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:D1")
    ws["A2"] = ("All variants share the same training config: Llama-3.2-3B policy + reward model, "
                "LoRA rank 32, GRPO via Tinker, KL penalty 0.2, 156 batches. Only the training "
                "corpus differs. Each data sheet holds one row per question × variant with the "
                "raw fields the evaluator recorded — no derived metrics.")
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A2:D2")
    ws.row_dimensions[2].height = 60

    headers = ["Variant", "Method", "Description", "Checkpoint (Tinker sampler URL)"]
    for i, h in enumerate(headers, 1):
        ws.cell(row=4, column=i, value=h)
    style_header_row(ws, 4, len(headers))

    checkpoints = {
        "v0-baseline-v2-sharma-methodology": "tinker://36cfb3db-6594-5a8a-be77-7d8ee4d755b5:train:0/sampler_weights/final",
        "v1a-naive-conservative":             "tinker://996b9011-dcb1-54b2-94b2-a070e275206e:train:0/sampler_weights/final",
        "v2a-insert-wei":                     "tinker://9074d4e6-df41-5b92-8293-9960ce75f263:train:0/sampler_weights/final",
        "v2b-insert-are-you-sure":            "tinker://7948b5f9-f77a-5d99-9be4-58d28fca6937:train:0/sampler_weights/final",
        "v2c-insert-answer":                  "tinker://dea56c63-3851-57e9-9e05-225506ab6348:train:0/sampler_weights/final",
        "v2d-insert-combined":                "tinker://16d6189c-7d6d-506d-a7bc-8c736cba712c:train:0/sampler_weights/final",
        "v3a-remove-plus-aysure":             "tinker://40ed48c2-3a9a-54d2-8aac-9e3c2829c83a:train:0/sampler_weights/final",
        "v3b-remove-plus-combined":           "tinker://d1f6a78b-65be-5f7c-9082-7bb83fea6372:train:0/sampler_weights/final",
    }

    for i, (stem, label, method, description) in enumerate(VARIANTS, start=5):
        ws.cell(row=i, column=1, value=label)
        ws.cell(row=i, column=2, value=method)
        ws.cell(row=i, column=3, value=description)
        ws.cell(row=i, column=4, value=checkpoints.get(stem, ""))
        for c in range(1, 5):
            ws.cell(row=i, column=c).border = BORDER_ALL
            ws.cell(row=i, column=c).alignment = Alignment(wrap_text=True, vertical="top")
        if stem.startswith("v0"):
            for c in range(1, 5):
                ws.cell(row=i, column=c).fill = BASELINE_FILL
                ws.cell(row=i, column=c).font = BASELINE_FONT
        ws.row_dimensions[i].height = 60

    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 46
    ws.column_dimensions["C"].width = 90
    ws.column_dimensions["D"].width = 78


def build_test_sheet(wb: Workbook, subdir: str, sheet_name: str,
                     jsonl_suffix: str | None,
                     column_specs: list[tuple[str, str, object]]) -> None:
    ws = wb.create_sheet(sheet_name)
    headers = ["variant"] + [h for h, _, _ in column_specs]
    for i, h in enumerate(headers, 1):
        ws.cell(row=1, column=i, value=h)
    style_header_row(ws, 1, len(headers))
    ws.row_dimensions[1].height = 32

    r = 2
    suffix = jsonl_suffix or ""
    for stem, label, _method, _desc in VARIANTS:
        path = HERE / subdir / f"{stem}{suffix}.jsonl"
        if not path.exists():
            continue
        is_baseline = stem.startswith("v0")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ws.cell(row=r, column=1, value=label)
                for c, (_hdr, field, idx) in enumerate(column_specs, start=2):
                    v = extract_value(rec, field, idx)
                    ws.cell(row=r, column=c, value=v)
                if is_baseline:
                    for c in range(1, len(headers) + 1):
                        ws.cell(row=r, column=c).fill = BASELINE_FILL
                r += 1

    # Widths: numeric cols narrow, text cols wide, question/response widest
    widths = [22]  # variant
    for hdr, _field, _idx in column_specs:
        if "reply" in hdr or "response" in hdr:
            widths.append(70)
        elif "question" in hdr or "text_head" in hdr:
            widths.append(50)
        elif "letter" in hdr or "condition" in hdr or "grade" in hdr or hdr == "apologised":
            widths.append(14)
        else:
            widths.append(18)
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    # Enable filter + freeze
    ws.auto_filter.ref = ws.dimensions
    ws.freeze_panes = "B2"


def main() -> None:
    wb = Workbook()
    del wb["Sheet"]
    build_readme(wb)
    for subdir, sheet_name, jsonl_suffix, column_specs in TESTS:
        build_test_sheet(wb, subdir, sheet_name, jsonl_suffix, column_specs)

    out = HERE / "all_raw_data.xlsx"
    wb.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
