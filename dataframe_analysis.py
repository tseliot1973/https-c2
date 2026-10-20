import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List

def find_csv_files(root: Path) -> List[Path]:
    # Ignore generated stats artifacts so reruns only process source CSV files.
    return sorted(p for p in root.rglob("*.csv") if not p.name.endswith("_stats.csv"))

def get_colors(n: int) -> List[str]:
    base = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
            "#1abc9c", "#e67e22", "#34495e", "#27ae60", "#c0392b"]
    return (base * (n // len(base) + 1))[:n]

def add_stat_box(ax, x_center: float, data: np.ndarray, label: str, color: str):
    if len(data) == 0:
        return
    stats = {
        "Min":    np.min(data),
        "Max":    np.max(data),
        "Mean":   np.mean(data),
        "Median": np.median(data),
        "Std":    np.std(data, ddof=0)
    }
    def fmt(v): 
        if abs(v) < 0.001 or abs(v) > 1e5: return f"{v:.2e}"
        if abs(v) < 0.1: return f"{v:.4f}"
        if abs(v) < 10:  return f"{v:.3f}"
        return f"{v:.2f}"

    text = "\n".join(f"{k:<6} {fmt(v)}" for k, v in stats.items())
    ax.text(
        x_center, -0.05, text,
        transform=ax.transAxes,
        fontsize=8.2, va='top', ha='center',
        bbox=dict(facecolor='white', edgecolor=color, linewidth=1.1,
                  boxstyle='round,pad=0.35', alpha=1.0),
        color='black'
    )

def main():
    parser = argparse.ArgumentParser(description="Violin + CDF comparison of numeric columns across CSV files")
    parser.add_argument("directory", type=str, help="Directory to search for .csv files")
    parser.add_argument("--labels", nargs="*", default=None, help="Custom labels (must match file count)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max_label_len", type=int, default=26, help="Max label length")
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory.")
        return

    csv_files = find_csv_files(root)
    if not csv_files:
        print("No .csv files found.")
        return

    n = len(csv_files)
    print(f"Found {n} CSV file{'s' if n > 1 else ''}:")
    for f in csv_files:
        print(f"  - {f}")

    # Labels
    if args.labels:
        if len(args.labels) != n:
            print(f"Error: {len(args.labels)} labels provided, {n} files found.")
            return
        labels = args.labels
    else:
        labels = [p.stem[:args.max_label_len-3] + "..." if len(p.stem) > args.max_label_len else p.stem
                  for p in csv_files]

    out_dir = Path(args.output_dir).resolve() if args.output_dir else root / "comparison_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dfs = []
    for p in csv_files:
        try:
            df = pd.read_csv(p)
            dfs.append(df)
            print(f"Loaded: {p.name} ({len(df):,} rows)")

            stats_path = p.with_name(f"{p.stem}_stats.csv")
            df.describe().to_csv(stats_path)
            print(f"  Saved stats: {stats_path.name}")
        except Exception as e:
            print(f"Failed to read {p.name}: {e}")
            return

    # Common numeric columns
    numeric_cols = set.intersection(*(set(df.select_dtypes(include=["float64", "int64"]).columns)
                                       for df in dfs))
    common_cols = sorted(numeric_cols)
    if not common_cols:
        print("No common numeric columns found.")
        return

    print(f"\nFound {len(common_cols)} common numeric columns.\n")

    colors = get_colors(n)

    for col in common_cols:
        datasets = []
        skip = False
        for df, lab in zip(dfs, labels):
            vals = df[col].dropna().values
            if len(vals) == 0:
                print(f"Skipping '{col}': no valid values in '{lab}'")
                skip = True
                break
            datasets.append(vals)
        if skip:
            continue

        # Violin plot
        fig, ax = plt.subplots(figsize=(max(8, n * 2.2), 7.8))
        data_long = pd.DataFrame({
            'value': np.concatenate(datasets),
            'dataset': np.concatenate([[lab] * len(d) for lab, d in zip(labels, datasets)])
        })

        sns.violinplot(
            data=data_long, x='dataset', y='value',
            hue='dataset', palette=colors, legend=False,
            ax=ax, saturation=0.68, width=0.82,
            inner='box', linewidth=1.0, cut=0
        )

        ax.set_title(f"Violin Plot — {col}", fontsize=16, pad=18)
        ax.set_ylabel(col, fontsize=13)
        ax.set_xlabel("Dataset", fontsize=13)
        ax.grid(axis="y", alpha=0.3, ls="--")
        plt.xticks(rotation=25, ha="right")
        plt.subplots_adjust(bottom=0.14 + 0.03 * (n // 3))

        for i, (data, lab, c) in enumerate(zip(datasets, labels, colors)):
            add_stat_box(ax, (i + 0.5) / n, data, lab, c)

        fig.savefig(out_dir / f"{col}_violin.png", dpi=320, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {col}_violin.png")

        # CDF
        fig, ax = plt.subplots(figsize=(11, 6.2))
        for d, lab, c in zip(datasets, labels, colors):
            x = np.sort(d)
            y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, label=lab, lw=2.2, c=c)

        ax.set_title(f"CDF — {col}", fontsize=15, pad=18)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Cumulative Probability", fontsize=12)
        ax.legend(fontsize=10, loc="best" if n > 4 else "lower right")
        ax.grid(alpha=0.25, ls="--")

        fig.savefig(out_dir / f"{col}_cdf.png", dpi=320, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {col}_cdf.png")

    print(f"\nAll plots saved to: {out_dir}")

if __name__ == "__main__":
    main()