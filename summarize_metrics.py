#!/usr/bin/env python3
"""
summarize_metrics.py

Summarize numeric columns (mean, std) from CSV/TXT results files like:

run,time,rmse_c0_imputation,rmse_c0_forecast,rmse_c0_overall,rmse_c1_imputation,rmse_c1_forecast,rmse_c1_overall
0,418.9731,0.5487,0.5702,0.5595,0.2773,0.2471,0.2627
1,425.8350,1.4446,1.4605,1.4525,0.7788,0.7668,0.7728
2,441.4203,1.1931,1.4868,1.3475,0.5421,0.6549,0.6010

By default it computes mean and std for all numeric columns EXCEPT 'run' and 'time'.
You can filter which columns to include via --include/--regex.
It prints a pretty table and also writes a CSV summary in the SAME folder as the input file.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def read_any(path: Path) -> pd.DataFrame:
    # Use Python engine with sep=None to auto-detect comma/space/tab
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        raise SystemExit(f"[ERROR] Failed to read {path}: {e}")

def choose_columns(df: pd.DataFrame, include=None, exclude=None, regex=False) -> list:
    # Start with numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Drop common non-metric numeric columns unless explicitly included
    default_drop = {"run", "epoch", "step", "iter", "iteration", "seconds", "wall_time"}
    # If user provided include, we honor that. Otherwise drop defaults.
    if include:
        if regex:
            import re
            keep = []
            for c in numeric_cols:
                if any(re.search(pat, c) for pat in include):
                    keep.append(c)
            numeric_cols = keep
        else:
            # include treats items as substrings
            keep = []
            for c in numeric_cols:
                if any(pat in c for pat in include):
                    keep.append(c)
            numeric_cols = keep
    else:
        numeric_cols = [c for c in numeric_cols if c.lower() not in default_drop]
    # Apply exclude last
    if exclude:
        if regex:
            import re
            numeric_cols = [c for c in numeric_cols if not any(re.search(pat, c) for pat in exclude)]
        else:
            numeric_cols = [c for c in numeric_cols if all(pat not in c for pat in exclude)]
    if not numeric_cols:
        raise SystemExit("[ERROR] No metric columns selected. "
                         "Try providing --include (optionally with --regex).")
    return numeric_cols

def summarize(df: pd.DataFrame, cols: list, precision: int = 6) -> pd.DataFrame:
    stats = df[cols].agg(['mean', 'std']).T.reset_index()
    stats = stats.rename(columns={'index': 'metric'})
    # Round for display; CSV will keep full precision unless --round_csv is set
    return stats.round(precision)

def main():
    ap = argparse.ArgumentParser(description="Summarize metrics (mean, std) from CSV/TXT.")
    ap.add_argument("files", nargs="+", type=Path, help="Input files (.csv/.txt)")
    ap.add_argument("--include", nargs="*", default=None,
                    help="Column filters to INCLUDE (regex if --regex, else substring). Example: --include rmse_ overall")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="Column filters to EXCLUDE (regex if --regex, else substring).")
    ap.add_argument("--regex", action="store_true",
                    help="Treat --include/--exclude patterns as regular expressions.")
    ap.add_argument("--precision", type=int, default=6,
                    help="Rounding precision for printed table (default: 6).")
    ap.add_argument("--round-csv", action="store_true",
                    help="Also round values in the CSV output (by --precision).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional path for the summary CSV. If multiple files are given and --out "
                         "is a directory, files will be written inside it. If omitted, the summary is "
                         "saved next to each input file as <name>_summary.csv.")
    args = ap.parse_args()

    combined_frames = []
    for f in args.files:
        df = read_any(f)
        cols = choose_columns(df, include=args.include, exclude=args.exclude, regex=args.regex)
        summary = summarize(df, cols, precision=args.precision)

        # Print a header and pretty table
        print(f"\n=== Summary for: {f} ===")
        print(f"(rows={len(df)}, metrics={len(cols)})")
        # Use pandas to_string for aligned console output
        print(summary.to_string(index=False))

        # Decide output path (default: SAME FOLDER as input)
        if args.out is None:
            out_path = f.with_suffix("")  # strip .txt/.csv
            out_path = out_path.parent / f"{out_path.name}_summary.csv"
        else:
            if args.out.is_dir() or (len(args.files) > 1 and not str(args.out).endswith('.csv')):
                args.out.mkdir(parents=True, exist_ok=True)
                out_path = args.out / f"{f.stem}_summary.csv"
            else:
                out_path = args.out

        csv_df = summary.copy()
        if not args.round_csv:
            # Recompute without rounding for CSV
            csv_df = df[cols].agg(['mean', 'std']).T.reset_index().rename(columns={'index': 'metric'})

        csv_df.to_csv(out_path, index=False)
        print(f"[saved] {out_path}")

        # For optional combined summary across multiple files
        summary['__source__'] = f.name
        combined_frames.append(summary)

    if len(combined_frames) > 1:
        # Wide-form combined summary: one table per file vertically stacked
        combo = pd.concat(combined_frames, ignore_index=True)
        print("\n=== Combined (stacked) summary across files ===")
        print(combo.to_string(index=False))
        # Optionally also pivot to show per-metric across files
        pivot = combo.pivot_table(index="metric", columns="__source__", values="mean")
        print("\n=== Means per metric (columns=file) ===")
        print(pivot.round(args.precision).to_string())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
