"""Generate a 4-panel bar chart comparing all 7 manipulation variants + baseline
across the 4 sycophancy evaluators. One subplot per evaluator; each subplot has
8 bars (baseline + 7 variants) using the natural metric scale.

Reads summary CSVs from ./<eval>/<variant>_summary.csv and writes comparison.png.
"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).parent

# Variant name -> short display label (order matters — left to right in each subplot)
VARIANTS = [
    ("v0-baseline-v2-sharma-methodology", "baseline"),
    ("v1a-naive-conservative",            "v1a\nremove"),
    ("v2a-insert-wei",                    "v2a\nWei"),
    ("v2b-insert-are-you-sure",           "v2b\nAreYouSure"),
    ("v2c-insert-answer",                 "v2c\nAnswer"),
    ("v2d-insert-combined",               "v2d\ncombined"),
    ("v3a-remove-plus-aysure",            "v3a\nrm+AY"),
    ("v3b-remove-plus-combined",          "v3b\nrm+comb"),
]

# (subdir, csv_metric_name, display_label, lower_is_better, scale, summary_suffix)
# scale=100: CSV stores as fraction (0.48), plot as percentage (48%).
# scale=1: CSV already in display units (25.6% or +1.34).
# summary_suffix: "" for the original summary CSV, "_llm" for LLM-regraded CSV.
PANELS = [
    ("are_you_sure_pushed", "ever_caved",                     "Pushed: 3-round cave rate (%)",                       True, 1,   ""),
    ("answer",              "sycophancy_from_assert_wrong",   "Answer: sycophancy from asserted wrong (%) — LLM judge", True, 100, "_llm"),
    ("are_you_sure",        "sycophancy_rate",                "Pure are-you-sure: sycophancy rate (%)*",             True, 100, ""),
    ("feedback",            "opinion_delta (like - dislike)", "Feedback: opinion delta (like − dislike) — LLM judge, 1-5", True, 1, "_llm"),
]


def read_metric(subdir: str, variant_file: str, metric: str, summary_suffix: str = "") -> float | None:
    """Look up a metric value in a summary CSV. Handles the various row shapes."""
    path = HERE / subdir / f"{variant_file}{summary_suffix}_summary.csv"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # feedback: coarse-pair-delta rows start with the metric name in col 0
            # answer/are_you_sure: rows with "metric,value,count,notes" — metric in col 0
            key = row[0].strip()
            if key == metric:
                val = row[1].strip()
                if val.endswith("%"):
                    return float(val.rstrip("%"))
                # Feedback deltas store as e.g. "+0.980" — strip the plus
                try:
                    return float(val.lstrip("+"))
                except ValueError:
                    return None
    return None


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    baseline_color = "#888888"
    bar_color = "#2a7fbf"
    improved_color = "#2ca02c"
    worsened_color = "#d62728"

    for ax, (subdir, metric, title, lower_better, scale, summary_suffix) in zip(axes, PANELS):
        labels = []
        values = []
        for variant_file, display in VARIANTS:
            v = read_metric(subdir, variant_file, metric, summary_suffix)
            labels.append(display)
            values.append((v if v is not None else 0.0) * scale)

        baseline_val = values[0]
        colors = []
        for i, v in enumerate(values):
            if i == 0 or abs(v - baseline_val) < 1e-9:
                colors.append(baseline_color)
                continue
            if lower_better:
                colors.append(improved_color if v < baseline_val else worsened_color)
            else:
                colors.append(improved_color if v > baseline_val else worsened_color)

        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(baseline_val, color="black", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        for bar, v in zip(bars, values):
            ax.annotate(
                f"{v:.2f}" if abs(v) < 5 else f"{v:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 3 if v >= 0 else -12),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    legend_handles = [
        mpatches.Patch(color=baseline_color, label="baseline (reference)"),
        mpatches.Patch(color=improved_color, label="less sycophantic than baseline"),
        mpatches.Patch(color=worsened_color, label="more sycophantic than baseline"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 0.98), frameon=False)
    fig.suptitle("Sycophancy manipulation variants — 4-evaluator comparison",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "* Pure are-you-sure baseline is already at floor (0%); differences across variants are near noise.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = HERE / "comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
