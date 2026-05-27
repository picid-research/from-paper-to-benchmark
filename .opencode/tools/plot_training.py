"""Generate a train/val loss curve PNG from a Lightning metrics.csv file.

Usage:
  uv run python .opencode/tools/plot_training.py \
    --metrics_csv artifacts/.../csv_logs/version_1/metrics.csv \
    --output_path artifacts/.../plots/loss_curves.png \
    [--early_stopping_epoch 47] \
    [--title "my_model — Loss Curves"]

Prints "PLOT_OK: <path>" on success or "PLOT_FAILED: <reason>" on error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


TRAIN_LOSS_COLUMNS = ("train/loss_epoch", "train_loss_epoch", "train/loss", "train_loss")
VAL_LOSS_COLUMNS = ("val/loss", "val_loss")


def first_existing(columns, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--early_stopping_epoch", type=int, default=None)
    parser.add_argument("--title", default="Training Loss Curves")
    args = parser.parse_args()

    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"PLOT_FAILED: missing dependency — {e}")
        sys.exit(0)

    csv_path = Path(args.metrics_csv)
    if not csv_path.exists():
        print(f"PLOT_FAILED: metrics_csv not found: {csv_path}")
        sys.exit(0)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"PLOT_FAILED: could not read CSV — {e}")
        sys.exit(0)

    train_col = first_existing(df.columns, TRAIN_LOSS_COLUMNS)
    val_col = first_existing(df.columns, VAL_LOSS_COLUMNS)

    if train_col is None and val_col is None:
        print(
            "PLOT_FAILED: no supported train/val loss columns found in "
            f"{csv_path}; tried {TRAIN_LOSS_COLUMNS + VAL_LOSS_COLUMNS}"
        )
        sys.exit(0)

    train_df = df[["epoch", train_col]].dropna(subset=[train_col]).reset_index(drop=True) if train_col else None
    val_df = df[["epoch", val_col]].dropna(subset=[val_col]).reset_index(drop=True) if val_col else None

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    if train_df is not None and not train_df.empty:
        ax.plot(train_df["epoch"], train_df[train_col], label="Train loss", color="#2196F3", linewidth=1.8)

    if val_df is not None and not val_df.empty:
        ax.plot(val_df["epoch"], val_df[val_col], label="Val loss", color="#FF5722", linewidth=1.8)

        best_idx = val_df[val_col].idxmin()
        best_epoch = val_df.loc[best_idx, "epoch"]
        best_val = val_df.loc[best_idx, val_col]
        ax.axvline(best_epoch, color="#FF5722", linestyle="--", linewidth=1.0, alpha=0.7, label=f"Best val (epoch {int(best_epoch)}, {best_val:.4f})")

    if args.early_stopping_epoch is not None:
        ax.axvline(args.early_stopping_epoch, color="#9C27B0", linestyle=":", linewidth=1.2, label=f"Early stop (epoch {args.early_stopping_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(args.title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
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
