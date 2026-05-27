"""Generate a horizontal bar chart comparing a new model against framework baselines.

Usage:
  uv run python .opencode/tools/plot_baselines.py \
    --input_json '{"baselines": {"MLP": 14.2, "CNN1D": 15.8}, "new_model_name": "MyModel", "new_model_value": 11.3}' \
    --metric_key "test/rmse_denormalized" \
    --lower_is_better true \
    --output_path vault/paper/plots/baseline_comparison.png \
    [--title "PRONOSTIA / prognostics — Baseline Comparison"]

input_json fields:
  baselines        dict[str, float]  model_name -> metric mean
  new_model_name   str
  new_model_value  float

Prints "PLOT_OK: <path>" on success or "PLOT_FAILED: <reason>" on error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _str_to_bool(v: str) -> bool:
    return v.lower() not in ("false", "0", "no")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--metric_key", required=True)
    parser.add_argument("--lower_is_better", default="true")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--title", default="Baseline Comparison")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"PLOT_FAILED: missing dependency — {e}")
        sys.exit(0)

    try:
        payload = json.loads(args.input_json)
    except json.JSONDecodeError as e:
        print(f"PLOT_FAILED: invalid JSON — {e}")
        sys.exit(0)

    baselines: dict[str, float] = payload.get("baselines", {})
    new_name: str = payload.get("new_model_name", "New Model")
    new_value: float | None = payload.get("new_model_value")

    if new_value is None:
        print("PLOT_FAILED: 'new_model_value' missing from input_json")
        sys.exit(0)

    all_models: dict[str, float] = {**baselines, new_name: new_value}
    lower_is_better = _str_to_bool(args.lower_is_better)

    sorted_models = sorted(all_models.items(), key=lambda x: x[1], reverse=not lower_is_better)
    names = [m[0] for m in sorted_models]
    values = [m[1] for m in sorted_models]

    colors = ["#2196F3" if n != new_name else "#FF5722" for n in names]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(names) + 1.5)))

    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    direction = "↓ lower is better" if lower_is_better else "↑ higher is better"
    ax.set_xlabel(f"{args.metric_key}  ({direction})")
    ax.set_title(args.title)
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FF5722", label=new_name),
        Patch(facecolor="#2196F3", label="Framework baselines"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    fig.tight_layout()

    try:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"PLOT_FAILED: could not save PNG — {e}")
        sys.exit(0)

    print(f"PLOT_OK: {output_path}")


if __name__ == "__main__":
    main()
