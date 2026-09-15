import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# 1. CONFIGURATION & FILE LIST
# ==============================================================================

# List of tuples: (model_data_file, model_name_for_graph, type)
MODEL_FILES = [
    ("v0-baseline-v2-sharma-methodology_llm_summary", "baseline", "baseline"),
    ("v1a-naive-conservative_llm_summary", "Remove", "Remove"),
    ("v2a-insert-wei_llm_summary", "insert\nWei", "insert"),
    ("v2b-insert-are-you-sure_llm_summary", "insert\nArithmetic", "insert"),
    ("v2c-insert-answer_llm_summary", "insert\nMisconceptions", "insert"),
    ("v2d-insert-combined_llm_summary", "insert\nCombined", "insert"),
    (
        "v3b-remove-plus-combined_llm_summary",
        "Remove +\ninsert Combined",
        "remove+insert",
    ),
]

# Add directory path and extension to file names
for i in range(len(MODEL_FILES)):
    new_name = "../Results/feedback/" + MODEL_FILES[i][0]
    if not new_name.endswith(".csv"):
        new_name = new_name + ".csv"
    MODEL_FILES[i] = (new_name, MODEL_FILES[i][1], MODEL_FILES[i][2])

OUTPUT_PDF_A = "feedback_res_graph_a_like_dislike.pdf"
OUTPUT_PDF_B = "feedback_res_graph_b_wrote_didnt_write.pdf"

# Dimensions and styling parameters
FIG_SIZE = (8.0, 3.2)
BAR_WIDTH = 0.35
FONT_SIZE = 8

# Color map for the model types
TYPE_COLOR_MAP = {
    "baseline": "#1f77b4",  # Blue
    "Remove": "#d62728",  # Red
    "insert": "#2ca02c",  # Green
    "remove+insert": "#ff7f0e",  # Orange
}


# ==============================================================================
# 2. DATA PARSING
# ==============================================================================
def load_metrics(file_path):
    """Reads summary CSV and extracts top/bottom deltas and total delta:

    Graph 1: (like - neutral, dislike - neutral, opinion_delta)
    Graph 2: (wrote - neutral, didnt_write - neutral, ownership_delta)
    """
    data = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    metric_name = row[0].strip()
                    val_str = row[1].strip()
                    try:
                        data[metric_name] = float(val_str)
                    except ValueError:
                        pass
    except Exception as e:
        print(
            f"ERROR: Failed to read file '{file_path}'. Details: {type(e).__name__} - {e}",
            file=sys.stderr,
        )
        raise

    try:
        like_delta = data["like - neutral"]
        dislike_delta = data["dislike - neutral"]
        wrote_delta = data["wrote - neutral"]
        didnt_write_delta = data["didnt_write - neutral"]

        opinion_delta = data.get(
            "opinion_delta (like - dislike)", like_delta - dislike_delta
        )
        ownership_delta = data.get(
            "ownership_delta (wrote - didnt_write)",
            wrote_delta - didnt_write_delta,
        )
    except KeyError as e:
        print(
            f"ERROR: Missing required delta metric {e} in '{file_path}'",
            file=sys.stderr,
        )
        raise

    return (like_delta, dislike_delta, opinion_delta), (
        wrote_delta,
        didnt_write_delta,
        ownership_delta,
    )


# ==============================================================================
# 3. GRAPH GENERATION
# ==============================================================================
def generate_pdf(
    labels,
    top_values,
    bottom_values,
    delta_values,
    types,
    y_label,
    filename,
    y_limits,
):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(len(labels))

    ax.set_ylim(y_limits)
    val_range = y_limits[1] - y_limits[0]

    # Neutral reference dashed line at Y = 0
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(
        len(labels) - 0.35,
        0,
        " neutral",
        va="center",
        ha="left",
        fontsize=FONT_SIZE,
        color="gray",
        fontweight="bold",
    )

    # Calculate uniform Y position for all bold delta labels near top of chart
    delta_y_pos = y_limits[1] - 0.08 * val_range

    # Baseline reference value for comparison (index 0)
    baseline_abs = abs(delta_values[0])

    for i in range(len(labels)):
        top_val = top_values[i]
        bot_val = bottom_values[i]
        delta_val = delta_values[i]
        color = TYPE_COLOR_MAP.get(types[i], "#1f77b4")

        y_min = min(top_val, bot_val)
        y_max = max(top_val, bot_val)
        height = max(y_max - y_min, 0.001)

        # Floating bar
        ax.bar(
            x[i],
            height,
            bottom=y_min,
            width=BAR_WIDTH,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=2,
        )

        # Labels attached to top and bottom of each bar
        top_str = f"{top_val:+.2f}"
        bot_str = f"{bot_val:+.2f}"
        offset = 0.03 * val_range

        ax.text(
            x[i],
            top_val + offset if top_val >= bot_val else top_val - offset,
            top_str,
            ha="center",
            va="bottom" if top_val >= bot_val else "top",
            fontsize=FONT_SIZE,
        )
        ax.text(
            x[i],
            bot_val - offset if bot_val <= top_val else bot_val + offset,
            bot_str,
            ha="center",
            va="top" if bot_val <= top_val else "bottom",
            fontsize=FONT_SIZE,
        )

        # Determine color based on comparison with baseline absolute value
        curr_abs = abs(delta_val)
        if i == 0 or curr_abs == baseline_abs:
            delta_color = "black"
        elif curr_abs > baseline_abs:
            delta_color = "red"
        else:
            delta_color = "green"

        # Vertically aligned bold delta value above each column
        delta_str = f"Δ{delta_val:+.2f}"
        ax.text(
            x[i],
            delta_y_pos,
            delta_str,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE,
            fontweight="bold",
            color=delta_color
        )

    # Axes styling
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE + 1)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)
    ax.set_xlim(-0.6, len(labels) - 0.1)

    # Clean border styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    model_labels = []
    model_types = []

    like_vals, dislike_vals, opinion_deltas = [], [], []
    wrote_vals, didnt_write_vals, ownership_deltas = [], [], []

    for path, label, m_type in MODEL_FILES:
        model_labels.append(label)
        model_types.append(m_type)

        (like_d, dislike_d, op_d), (wrote_d, didnt_write_d, ow_d) = (
            load_metrics(path)
        )

        like_vals.append(like_d)
        dislike_vals.append(dislike_d)
        opinion_deltas.append(op_d)

        wrote_vals.append(wrote_d)
        didnt_write_vals.append(didnt_write_d)
        ownership_deltas.append(ow_d)

    # Compute global minimum and maximum across BOTH datasets
    all_metrics = (
        like_vals + dislike_vals + wrote_vals + didnt_write_vals + [0.0]
    )
    global_min, global_max = min(all_metrics), max(all_metrics)
    global_range = global_max - global_min if global_max != global_min else 1.0

    # Expand top margin to comfortably fit the vertically aligned bold deltas
    global_y_limits = (
        global_min - 0.25 * global_range,
        global_max + 0.35 * global_range,
    )

    # Graph 1: Like vs Dislike Deltas
    generate_pdf(
        labels=model_labels,
        top_values=like_vals,
        bottom_values=dislike_vals,
        delta_values=opinion_deltas,
        types=model_types,
        y_label="Mean Positivity Score\nChange from Neutral",
        filename=OUTPUT_PDF_A,
        y_limits=global_y_limits,
    )

    # Graph 2: Wrote vs Didn't Write Deltas
    generate_pdf(
        labels=model_labels,
        top_values=wrote_vals,
        bottom_values=didnt_write_vals,
        delta_values=ownership_deltas,
        types=model_types,
        y_label="Mean Positivity Score\nChange from Neutral",
        filename=OUTPUT_PDF_B,
        y_limits=global_y_limits,
    )

    print(
        f"Successfully generated PDF graphs:\n - {OUTPUT_PDF_A}\n - {OUTPUT_PDF_B}"
    )