import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# graph = "results" # comparing Llama under different manipulations
graph = "validation" # method validation - comparing to other LLMs

# ==============================================================================
# 1. CONFIGURATION & FILE LIST
# ==============================================================================
# List of tuples: (model_data_file, model_name_for_graph, type)
if graph == "results":
    MODEL_FILES = [
        ("v0-baseline-v2-sharma-methodology_summary", "baseline", "baseline"),
        ("v1a-naive-conservative_summary", "Remove", "Remove"),
        ("v2a-insert-wei_summary", "insert\nWei", "insert"),
        ("v2b-insert-are-you-sure_summary", "insert\nArithmetic", "insert"),
        ("v2c-insert-answer_summary", "insert\nMisconceptions", "insert"),
        ("v2d-insert-combined_summary", "insert\nCombined", "insert"),
        ("v3b-remove-plus-combined_summary", "Remove +\ninsert Combined", "remove+insert"),
    ]
elif graph == "validation":
    MODEL_FILES = [
        ("v0-baseline-v2-sharma-methodology_summary", "Our baseline (Llama-3.2-3B)", "baseline"),
        ("openrouter-gpt-3.5-turbo_summary", "GPT-3.5-turbo", "Remove"),
        ("openrouter-llama-3.1-70b-instruct_summary", "llama-3.1-70b-instruct", "insert"),
        ("openrouter-llama-3.2-3b-instruct_summary.csv", "llama-3.2-3b-instruct", "insert")
    ]

# Add path to file names
FILE_DIR = str(Path(__file__).resolve().parent.parent)
for i in range(len(MODEL_FILES)):
    new_name = FILE_DIR + "/Results/are_you_sure_pushed/" + MODEL_FILES[i][0]
    if not new_name.endswith(".csv"):
        new_name = new_name + ".csv"
    MODEL_FILES[i] = (new_name, MODEL_FILES[i][1], MODEL_FILES[i][2])

if graph == "results":
    OUTPUT_PDF = FILE_DIR + "/Graphs/AreYouSure_pushed_res_graph_all_rounds.pdf"
elif graph == "validation":
    OUTPUT_PDF = FILE_DIR + "/Graphs/AreYouSure_pushed_method-val_graph_all_rounds.pdf"


# Standard dimensions (width=8.5 in, height=2.8 in)
FIG_SIZE = (8.5, 2.8)
BAR_WIDTH = 0.22
BAR_SPACING = 0.03
FONT_SIZE = 8

# Color map for the 4 model types
TYPE_COLOR_MAP = {
    "baseline": "#1f77b4",       # Blue
    "Remove": "#d62728",         # Red
    "insert": "#2ca02c",         # Green
    "remove+insert": "#ff7f0e",  # Orange
}


# ==============================================================================
# 2. DATA PARSING
# ==============================================================================
def load_metrics(file_path):
    """Reads CSV and extracts (value, count_str) tuples for r1, r2, r3."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(
            f"ERROR: Failed to read file '{file_path}'. Details: {type(e).__name__} - {e}",
            file=sys.stderr,
        )
        raise

    if "metric" in df.columns:
        df["metric"] = df["metric"].astype(str).str.strip()

    def extract_val_and_count(metric_name, fallback_row_idx):
        row = (
            df[df["metric"] == metric_name]
            if "metric" in df.columns
            else pd.DataFrame()
        )
        if row.empty and len(df) > fallback_row_idx:
            row = df.iloc[[fallback_row_idx]]

        if row.empty or "value" not in row.columns:
            raise KeyError(
                f"Metric '{metric_name}' (or row index {fallback_row_idx}) not found in file"
            )

        val_raw = row["value"].values[0]
        if isinstance(val_raw, str):
            val_raw = val_raw.strip().rstrip("%")
        val = float(val_raw)
        val_pct = val * 100.0 if val <= 1.0 else val

        count_str = ""
        if "count" in row.columns and pd.notna(row["count"].values[0]):
            count_str = str(row["count"].values[0])

        return val_pct, count_str

    try:
        # Row 8: cumulative_caved_by_r1 (index 6 in 0-based dataframe)
        r1_data = extract_val_and_count("cumulative_caved_by_r1", 6)
        # Row 9: cumulative_caved_by_r2 (index 7 in 0-based dataframe)
        r2_data = extract_val_and_count("cumulative_caved_by_r2", 7)
        # Row 10: cumulative_caved_by_r3 (index 8 in 0-based dataframe)
        r3_data = extract_val_and_count("cumulative_caved_by_r3", 8)
    except Exception as e:
        print(
            f"ERROR: Failed to extract metrics from '{file_path}'. Details: {type(e).__name__} - {e}",
            file=sys.stderr,
        )
        raise

    return r1_data, r2_data, r3_data


model_labels = []
model_types = []
r1_vals, r1_counts = [], []
r2_vals, r2_counts = [], []
r3_vals, r3_counts = [], []

for path, label, m_type in MODEL_FILES:
    model_labels.append(label)
    model_types.append(m_type)

    (r1_v, r1_c), (r2_v, r2_c), (r3_v, r3_c) = load_metrics(path)

    r1_vals.append(r1_v)
    r1_counts.append(r1_c)

    r2_vals.append(r2_v)
    r2_counts.append(r2_c)

    r3_vals.append(r3_v)
    r3_counts.append(r3_c)


# ==============================================================================
# 3. GRAPH GENERATION
# ==============================================================================
def generate_grouped_pdf(labels, r1_v, r1_c, r2_v, r2_c, r3_v, r3_c, types, filename):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(len(labels))

    # Plot grouped bars (r1, r2, r3) per model
    for i, m_type in enumerate(types):
        base_color = TYPE_COLOR_MAP.get(m_type, "#1f77b4")

        # Round 1 (r1)
        b1 = ax.bar(
            x[i] - BAR_WIDTH - BAR_SPACING,
            r1_v[i],
            width=BAR_WIDTH,
            color=base_color,
            alpha=0.45,
        )
        # Round 2 (r2)
        b2 = ax.bar(
            x[i],
            r2_v[i],
            width=BAR_WIDTH,
            color=base_color,
            alpha=0.75,
        )
        # Round 3 (r3)
        b3 = ax.bar(
            x[i] + BAR_WIDTH + BAR_SPACING,
            r3_v[i],
            width=BAR_WIDTH,
            color=base_color,
            alpha=1.00,
        )

        # Labels on top of each bar
        ax.bar_label(b1, labels=[r1_c[i]], padding=2, fontsize=FONT_SIZE - 2, rotation=90)
        ax.bar_label(b2, labels=[r2_c[i]], padding=2, fontsize=FONT_SIZE - 2, rotation=90)
        ax.bar_label(b3, labels=[r3_c[i]], padding=2, fontsize=FONT_SIZE - 2, rotation=90)

    ax.set_ylim(0, 115)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Frequency of\nsycophantic\nresponse (%)", fontsize=FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE)

    # Legend indicating the opacity level for each round
    legend_elements = [
        Patch(facecolor="grey", alpha=0.45, label="r1 (caved by round 1)"),
        Patch(facecolor="grey", alpha=0.75, label="r2 (caved by round 2)"),
        Patch(facecolor="grey", alpha=1.00, label="r3 (caved by round 3)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=FONT_SIZE, framealpha=0.8)

    if graph == "validation":
        ax.set_xlim(right=ax.get_xlim()[1] + 1)  # validation bars are high, overlapped with legend
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig)


# Output single PDF with 3 grouped bars per model label
generate_grouped_pdf(
    model_labels,
    r1_vals,
    r1_counts,
    r2_vals,
    r2_counts,
    r3_vals,
    r3_counts,
    model_types,
    OUTPUT_PDF,
)