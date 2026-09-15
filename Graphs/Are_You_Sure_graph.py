import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

graph = "results" # comparing Llama under different manipulations
# graph = "validation" # method validation - comparing to other LLMs

# ==============================================================================
# 1. CONFIGURATION & FILE LIST
# ==============================================================================

# List of tuples: (model_data_file, model_name_for_graph, type)
if graph == "results":
    MODEL_FILES = [
        ("v0-baseline-v2-sharma-methodology_summary", "baseline", "baseline"),
        ("v1a-naive-conservative_summary", "Remove", "Remove"),
        ("v2a-insert-wei_summary", "insert\nWei", "insert"),
        ("v2b-insert-are-you-sure_summary", "insert\nArith.", "insert"),
        ("v2c-insert-answer_summary", "insert\nMisconcep.", "insert"),
        ("v2d-insert-combined_summary", "insert\nCombined", "insert"),
        ("v3b-remove-plus-combined_summary", "Remove +\ninsert Comb.", "remove+insert"),
    ]
elif graph == "validation":
    MODEL_FILES = [
        ("v0-baseline-v2-sharma-methodology_summary", "Our baseline\n(Llama-3.2-3B)", "baseline"),
        ("openrouter-gpt-3.5-turbo_summary", "GPT-3.5\n-turbo", "Remove"),
        ("openrouter-llama-3.1-70b-instruct-v2-regraded_summary", "llama-3.1\n-70b-instruct", "insert"),
        ("openrouter-llama-3.2-3b-instruct_summary.csv", "llama-3.2\n-3b-instruct", "insert")
    ]

## add path to file names
FILE_DIR = str(Path(__file__).resolve().parent.parent)
for i in range(len(MODEL_FILES)):
    new_name = FILE_DIR + "/Results/are_you_sure/" + MODEL_FILES[i][0]
    if not new_name.endswith(".csv"):
        new_name = new_name + ".csv"
    MODEL_FILES[i] = \
        (new_name,
         MODEL_FILES[i][1], MODEL_FILES[i][2])

if graph == "results":
    OUTPUT_PDF_A = FILE_DIR + "/Graphs/AreYouSure_res_graph_a_apologetic.pdf"
    OUTPUT_PDF_B = FILE_DIR + "/Graphs/AreYouSure_res_graph_b_sycophancy.pdf"
elif graph == "validation":
    OUTPUT_PDF_A = FILE_DIR + "/Graphs/AreYouSure_method-val_graph_a_apologetic.pdf"
    OUTPUT_PDF_B = FILE_DIR + "/Graphs/AreYouSure_method-val_graph_b_sycophancy.pdf"


# Standard dimensions shared across both graphs
if graph == "results":
    FIG_SIZE = (4.0, 1.5)
    BAR_WIDTH = 0.3
    FONT_SIZE = 5
elif graph == "validation":
    FIG_SIZE = (4.0, 2.5)
    BAR_WIDTH = 0.3
    FONT_SIZE = 7

# Color map for the 4 model types
TYPE_COLOR_MAP = {
    "baseline": "#1f77b4",  # Blue
    "Remove": "#d62728",  # Red
    "insert": "#2ca02c",  # Green
    "remove+insert": "#ff7f0e",  # Orange
}


# ==============================================================================
# DATA PARSING
# ==============================================================================
def load_metrics(file_path):
    """Reads CSV and extracts (value, count_str) tuples for both metrics."""
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

        val = float(row["value"].values[0])
        val_pct = val * 100.0 if val <= 1.0 else val

        count_str = ""
        if "count" in row.columns and pd.notna(row["count"].values[0]):
            count_str = str(row["count"].values[0])

        return val_pct, count_str

    try:
        sycophancy_data = extract_val_and_count("sycophancy_rate", 6)
        apologetic_data = extract_val_and_count("apologetic_when_correct", 7)
    except Exception as e:
        print(
            f"ERROR: Failed to extract metrics from '{file_path}'. Details: {type(e).__name__} - {e}",
            file=sys.stderr,
        )
        raise

    return apologetic_data, sycophancy_data


model_labels = []
model_types = []
apologetic_vals, apologetic_counts = [], []
sycophancy_vals, sycophancy_counts = [], []

for path, label, m_type in MODEL_FILES:
    model_labels.append(label)
    model_types.append(m_type)

    (apol_val, apol_cnt), (syco_val, syco_cnt) = load_metrics(path)

    apologetic_vals.append(apol_val)
    apologetic_counts.append(apol_cnt)

    sycophancy_vals.append(syco_val)
    sycophancy_counts.append(syco_cnt)


# ==============================================================================
# GRAPH GENERATION
# ==============================================================================
def generate_pdf(labels, values, count_labels, types, filename):
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Map each type to its corresponding color
    bar_colors = [TYPE_COLOR_MAP.get(t, "#1f77b4") for t in types]

    bars = ax.bar(labels, values, width=BAR_WIDTH, color=bar_colors)

    if graph == "validation":
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 50, 100])
    elif graph == "results":
        ax.set_ylim(0, 65)
        ax.set_yticks([0, 50])
    ax.set_ylabel("Frequency of\nsycophantic\nresponse (%)")
    if graph == "results":
        ax.set_ylabel("Frequency of\nsycophantic\nresponse (%)",fontsize=FONT_SIZE)

    ax.bar_label(bars, labels=count_labels, padding=3, fontsize=FONT_SIZE)
    ax.tick_params(axis="x", labelsize=FONT_SIZE)

    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig)


# Output Graph (a) and Graph (b) PDFs with colored bars per type
generate_pdf(
    model_labels, apologetic_vals, apologetic_counts, model_types, OUTPUT_PDF_A
)
generate_pdf(
    model_labels, sycophancy_vals, sycophancy_counts, model_types, OUTPUT_PDF_B
)