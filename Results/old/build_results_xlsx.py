"""Build a single XLSX with all evaluation results laid out for reading.

Sheets:
  1. README         — one row per manipulation, method + plain-English description
  2. pushed         — are_you_sure_pushed metrics (3-round escalating pressure)
  3. are_you_sure   — Sharma 4-turn "are you sure?" metrics (pure challenge)
  4. answer         — single-turn hint-uptake metrics
  5. feedback       — positivity-shift metrics on rhetorical arguments

Reads summary CSVs from ./<eval>/<variant>_summary.csv.
"""
from pathlib import Path
import csv

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

HERE = Path(__file__).parent

# (file_stem, short_label, method, description)
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
     "Added ~5,000 synthetic anti-sycophancy pairs from Wei et al. (2024). "
     "Targets feedback-style positivity shifts."),
    ("v2b-insert-are-you-sure", "v2b — AreYouSure",
     "insert-are-you-sure",
     "Added ~5,000 synthetic anti-sycophancy pairs on math questions with are-you-sure structure. "
     "Targets multi-turn pushed capitulation."),
    ("v2c-insert-answer", "v2c — Answer",
     "insert-answer",
     "Added ~5,000 synthetic anti-sycophancy pairs on common-misconception answer format. "
     "Targets single-turn asserted-wrong sycophancy."),
    ("v2d-insert-combined", "v2d — combined",
     "insert-combined",
     "Added ~5,000 synthetic pairs combining Wei + AreYouSure + Answer into a single pool."),
    ("v3a-remove-plus-aysure", "v3a — remove + AreYouSure",
     "remove-naive-conservative + insert-are-you-sure",
     "First applied the v1a removal filter (~4.6% pairs removed), then appended v2b's AreYouSure pairs."),
    ("v3b-remove-plus-combined", "v3b — remove + combined",
     "remove-naive-conservative + insert-combined",
     "First applied the v1a removal filter, then appended the combined synthetic set (Wei + AreYouSure + Answer)."),
]


def read_summary_rows(subdir: str, variant_stem: str) -> dict[str, list[str]]:
    """Return {row_key: full_row_as_list} from a summary CSV.

    row_key is column 0 of each row (metric name or condition name). This lets
    callers pick which subsequent column they want — needed because per-condition
    rows have multiple numeric columns (n, pct_correct, pct_incorrect_hint, ...).
    """
    path = HERE / subdir / f"{variant_stem}_summary.csv"
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            key = row[0].strip()
            if key and not key.startswith("---") and not key.startswith('"---'):
                out[key] = [c.strip() for c in row]
    return out


def to_number(s: str):
    """Convert '25.6%' or '+0.980' or '13/87' or '199' to a float, or return original."""
    if s.endswith("%"):
        try:
            return float(s.rstrip("%")) / 100
        except ValueError:
            return s
    try:
        return float(s.lstrip("+"))
    except ValueError:
        return s


# ---- styling helpers ----------------------------------------------------- #
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


def autosize(ws, min_width: int = 10, max_width: int = 60) -> None:
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        longest = max(
            (len(str(c.value)) for c in col if c.value is not None),
            default=min_width,
        )
        ws.column_dimensions[col_letter].width = min(max(longest + 2, min_width), max_width)


# ---- sheet builders ------------------------------------------------------- #
def build_readme(wb: Workbook) -> None:
    ws = wb.create_sheet("README", 0)
    ws["A1"] = "Sycophancy manipulation experiments — results"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:D1")
    ws["A2"] = ("Each row below is one training variant. All variants share the same training config: "
                "Llama-3.2-3B policy + reward model, LoRA rank 32, GRPO via Tinker, KL penalty 0.2, 156 batches. "
                "Only the training corpus differs.")
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A2:D2")
    ws.row_dimensions[2].height = 45

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

    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 44
    ws.column_dimensions["C"].width = 90
    ws.column_dimensions["D"].width = 78

    ws.cell(row=len(VARIANTS) + 7, column=1, value="Evaluator sheets").font = Font(bold=True, size=12)
    eval_notes = [
        ("pushed",       "are_you_sure_pushed — 4-turn structure with 3 escalating rounds of pressure toward a specific wrong-letter alternative. Headline: `ever_caved` (fraction of initially-correct answers the model eventually changed to the pushed wrong alternative)."),
        ("are_you_sure", "Sharma-methodology pure 4-turn 'are you sure?' challenge (no alternative pushed). Headline: `sycophancy_rate` (fraction of initially-correct answers that flipped to wrong) and `apologetic_when_correct` (T4 contained apology language even if final answer stayed correct)."),
        ("answer",       "Single-turn trivia questions with a hint under 4 framings (neutral / reject_correct / assert_wrong / assert_correct). Headline: `sycophancy_from_assert_wrong` (was correct in neutral, then wrong when user asserted a specific wrong answer)."),
        ("feedback",     "Model comments on a rhetorical argument shown in 5 framings (neutral / like / dislike / wrote / didnt_write). Positivity scored via keyword lexicon. Headline: `opinion_delta` (like−dislike) and `ownership_delta` (wrote−didnt_write)."),
    ]
    for j, (name, note) in enumerate(eval_notes):
        r = len(VARIANTS) + 8 + j
        ws.cell(row=r, column=1, value=name).font = Font(bold=True)
        ws.cell(row=r, column=2, value=note)
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
        ws.cell(row=r, column=2).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r].height = 45


def build_metric_sheet(wb: Workbook, sheet_name: str, subdir: str,
                       title: str, metric_order: list[tuple[str, int, str]]) -> None:
    """metric_order: list of (row_key, column_index_in_that_row, display_header).

    column_index=1 is the standard "value" column for "metric,value,count" rows.
    For per-condition rows use the index of the desired sub-metric (e.g. 5 for
    pct_correct in answer.jsonl summaries).
    """
    ws = wb.create_sheet(sheet_name)
    ws["A1"] = title
    ws["A1"].font = TITLE_FONT
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(metric_order) + 1)

    display_headers = ["Variant"] + [h for _, _, h in metric_order]
    for i, h in enumerate(display_headers, 1):
        ws.cell(row=3, column=i, value=h)
    style_header_row(ws, 3, len(display_headers))
    ws.row_dimensions[3].height = 40

    for r, (stem, label, _method, _desc) in enumerate(VARIANTS, start=4):
        rows = read_summary_rows(subdir, stem)
        ws.cell(row=r, column=1, value=label)
        for c, (row_key, col_idx, _hdr) in enumerate(metric_order, start=2):
            row_data = rows.get(row_key, [])
            raw = row_data[col_idx].strip() if col_idx < len(row_data) else ""
            val = to_number(raw) if raw else ""
            cell = ws.cell(row=r, column=c, value=val)
            if isinstance(val, float):
                cell.number_format = "0.000" if abs(val) < 10 else "0.0"
            cell.alignment = Alignment(horizontal="center")
        for c in range(1, len(display_headers) + 1):
            ws.cell(row=r, column=c).border = BORDER_ALL
        if stem.startswith("v0"):
            for c in range(1, len(display_headers) + 1):
                ws.cell(row=r, column=c).fill = BASELINE_FILL
                ws.cell(row=r, column=c).font = BASELINE_FONT

    autosize(ws, min_width=14, max_width=32)
    ws.freeze_panes = "B4"


def main() -> None:
    wb = Workbook()
    del wb["Sheet"]  # remove default sheet
    build_readme(wb)

    # Pushed & pure are_you_sure: all rows are "metric,value,count,notes" so col 1
    build_metric_sheet(
        wb, "pushed", "are_you_sure_pushed",
        "are_you_sure_pushed — 3-round escalating pressure toward a specific wrong-letter alternative",
        [
            ("total_questions",             1, "total_questions"),
            ("extraction_ok",               1, "extraction_ok"),
            ("extraction_failed",           1, "extraction_failed"),
            ("initial_accuracy",            1, "initial_accuracy"),
            ("cumulative_caved_by_r1",      1, "cumulative_caved_by_r1"),
            ("cumulative_caved_by_r2",      1, "cumulative_caved_by_r2"),
            ("cumulative_caved_by_r3",      1, "cumulative_caved_by_r3"),
            ("ever_caved",                  1, "ever_caved"),
            ("final_answer_is_challenge_Y", 1, "final_answer_is_challenge_Y"),
            ("final_answer_still_correct",  1, "final_answer_still_correct"),
        ],
    )
    build_metric_sheet(
        wb, "are_you_sure", "are_you_sure",
        "are_you_sure (pure Sharma challenge, no alternative)",
        [
            ("total_questions",         1, "total_questions"),
            ("extraction_ok",           1, "extraction_ok"),
            ("extraction_failed",       1, "extraction_failed"),
            ("initial_accuracy",        1, "initial_accuracy"),
            ("sycophancy_rate",         1, "sycophancy_rate"),
            ("apologetic_when_correct", 1, "apologetic_when_correct"),
            ("held_firm_rate",          1, "held_firm_rate"),
            ("self_correction_rate",    1, "self_correction_rate"),
            ("initial_correct__apologised__final_correct", 1, "initial_correct__apologised__final_correct"),
            ("initial_correct__apologised__final_wrong",   1, "initial_correct__apologised__final_wrong"),
            ("initial_correct__no_apology__final_correct", 1, "initial_correct__no_apology__final_correct"),
            ("initial_correct__no_apology__final_wrong",   1, "initial_correct__no_apology__final_wrong"),
            ("initial_wrong__apologised__final_correct",   1, "initial_wrong__apologised__final_correct"),
            ("initial_wrong__apologised__final_wrong",     1, "initial_wrong__apologised__final_wrong"),
            ("initial_wrong__no_apology__final_correct",   1, "initial_wrong__no_apology__final_correct"),
            ("initial_wrong__no_apology__final_wrong",     1, "initial_wrong__no_apology__final_wrong"),
        ],
    )
    # Answer: per-condition rows have cols (condition, n, n_correct, n_incorrect_hint,
    # n_other, pct_correct, pct_incorrect_hint) — use cols 1..6 to expose everything.
    build_metric_sheet(
        wb, "answer", "answer",
        "answer — single-turn trivia with 4 hint framings",
        [
            ("neutral",        1, "neutral.n"),
            ("neutral",        2, "neutral.n_correct"),
            ("neutral",        3, "neutral.n_incorrect_hint"),
            ("neutral",        4, "neutral.n_other"),
            ("neutral",        5, "neutral.pct_correct"),
            ("neutral",        6, "neutral.pct_incorrect_hint"),
            ("reject_correct", 5, "reject_correct.pct_correct"),
            ("reject_correct", 6, "reject_correct.pct_incorrect_hint"),
            ("assert_wrong",   5, "assert_wrong.pct_correct"),
            ("assert_wrong",   6, "assert_wrong.pct_incorrect_hint"),
            ("assert_correct", 5, "assert_correct.pct_correct"),
            ("assert_correct", 6, "assert_correct.pct_incorrect_hint"),
            ("sycophancy_from_reject_correct", 1, "sycophancy_from_reject_correct"),
            ("sycophancy_from_assert_wrong",   1, "sycophancy_from_assert_wrong"),
            ("lift_from_assert_correct",       1, "lift_from_assert_correct"),
            ("caves_to_wrong_hint_rate",       1, "caves_to_wrong_hint_rate"),
        ],
    )
    # Feedback: per-condition rows have cols (condition, n, mean_net_score,
    # mean_positive_words, mean_negative_words). Delta rows have cols
    # (metric, mean_delta, frac_args_positive, n_args, notes).
    build_metric_sheet(
        wb, "feedback", "feedback",
        "feedback — argument-positivity shift across 5 framings; deltas paired per argument",
        [
            ("neutral",     1, "neutral.n"),
            ("neutral",     2, "neutral.mean_net_score"),
            ("neutral",     3, "neutral.mean_positive_words"),
            ("neutral",     4, "neutral.mean_negative_words"),
            ("like",        2, "like.mean_net_score"),
            ("dislike",     2, "dislike.mean_net_score"),
            ("wrote",       2, "wrote.mean_net_score"),
            ("didnt_write", 2, "didnt_write.mean_net_score"),
            ("like - neutral",        1, "like - neutral.mean_delta"),
            ("like - neutral",        2, "like - neutral.frac_args_positive"),
            ("dislike - neutral",     1, "dislike - neutral.mean_delta"),
            ("dislike - neutral",     2, "dislike - neutral.frac_args_positive"),
            ("wrote - neutral",       1, "wrote - neutral.mean_delta"),
            ("wrote - neutral",       2, "wrote - neutral.frac_args_positive"),
            ("didnt_write - neutral", 1, "didnt_write - neutral.mean_delta"),
            ("didnt_write - neutral", 2, "didnt_write - neutral.frac_args_positive"),
            ("opinion_delta (like - dislike)",         1, "opinion_delta (like - dislike)"),
            ("opinion_delta (like - dislike)",         2, "opinion_delta.frac_args_positive"),
            ("ownership_delta (wrote - didnt_write)",  1, "ownership_delta (wrote - didnt_write)"),
            ("ownership_delta (wrote - didnt_write)",  2, "ownership_delta.frac_args_positive"),
        ],
    )

    out = HERE / "all_results.xlsx"
    wb.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
