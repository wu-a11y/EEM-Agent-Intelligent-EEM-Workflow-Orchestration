# -*- coding: utf-8 -*-
"""
Batch set EEM cells to 0 by Ex/Em thresholds.

Default layout (matches this repo's EEM xlsx exports):
- Ex wavelengths are in the first row (row 0), columns 1..N
- Em wavelengths are in the first column (col 0), rows 1..M
- Data matrix is rows 1..M, cols 1..N

Rule (default, inclusive):
- Ex >= 500  -> entire corresponding columns set to 0
- Em <= 300  -> entire corresponding rows set to 0

This script writes to a new output directory by default, to avoid overwriting raw files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _to_float_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _zero_eem_df(
    df: pd.DataFrame,
    *,
    ex_threshold: float,
    em_threshold: float,
    inclusive: bool,
    ex_axis: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns (modified_df, stats).
    ex_axis: "cols" (Ex in first row) or "rows" (Ex in first column).
    """
    if df.shape[0] < 2 or df.shape[1] < 2:
        return df, {
            "ex_zero_count": 0,
            "em_zero_count": 0,
            "changed_cells": 0,
        }

    data = df.iloc[1:, 1:].to_numpy(copy=True)
    before = None
    try:
        before = (data != 0).sum()
    except Exception:
        before = None

    if inclusive:
        ex_cmp = np.greater_equal
        em_cmp = np.less_equal
    else:
        ex_cmp = np.greater
        em_cmp = np.less

    if ex_axis == "cols":
        ex_vals = _to_float_array(df.iloc[0, 1:])
        em_vals = _to_float_array(df.iloc[1:, 0])

        ex_zero_cols = ex_cmp(ex_vals, ex_threshold) & ~np.isnan(ex_vals)
        em_zero_rows = em_cmp(em_vals, em_threshold) & ~np.isnan(em_vals)

        if ex_zero_cols.any():
            data[:, ex_zero_cols] = 0
        if em_zero_rows.any():
            data[em_zero_rows, :] = 0

        ex_zero_count = int(ex_zero_cols.sum())
        em_zero_count = int(em_zero_rows.sum())
    elif ex_axis == "rows":
        ex_vals = _to_float_array(df.iloc[1:, 0])
        em_vals = _to_float_array(df.iloc[0, 1:])

        ex_zero_rows = ex_cmp(ex_vals, ex_threshold) & ~np.isnan(ex_vals)
        em_zero_cols = em_cmp(em_vals, em_threshold) & ~np.isnan(em_vals)

        if ex_zero_rows.any():
            data[ex_zero_rows, :] = 0
        if em_zero_cols.any():
            data[:, em_zero_cols] = 0

        ex_zero_count = int(ex_zero_rows.sum())
        em_zero_count = int(em_zero_cols.sum())
    else:
        raise ValueError("ex_axis must be 'cols' or 'rows'")

    df_out = df.copy()
    df_out.iloc[1:, 1:] = data

    changed_cells = 0
    if before is not None:
        try:
            after = (data != 0).sum()
            changed_cells = int(before - after)
        except Exception:
            changed_cells = 0

    return df_out, {
        "ex_zero_count": ex_zero_count,
        "em_zero_count": em_zero_count,
        "changed_cells": changed_cells,
    }


def _default_output_dir(input_dir: Path) -> Path:
    return input_dir.with_name(f"{input_dir.name}_zeroed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".." / "data" / "raw",
        help="Directory containing EEM .xlsx files (default: ../data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write updated .xlsx files (default: <input>_zeroed)",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.xlsx",
        help="Which files to process (default: *.xlsx)",
    )
    parser.add_argument("--ex-threshold", type=float, default=500.0)
    parser.add_argument("--em-threshold", type=float, default=300.0)
    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="Use Ex > threshold and Em < threshold (default is inclusive: >= / <=).",
    )
    parser.add_argument(
        "--ex-axis",
        choices=["cols", "rows"],
        default="cols",
        help="Where Ex wavelengths are stored: first row -> cols (default) or first column -> rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output dir.",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    output_dir: Path = args.output_dir or _default_output_dir(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.glob))
    if not files:
        print(f"No files matched: {input_dir / args.glob}")
        return 0

    inclusive = not args.exclusive
    total_changed = 0
    for in_path in files:
        out_path = output_dir / in_path.name
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {in_path.name} (exists)")
            continue

        try:
            sheets: dict[str, pd.DataFrame] = pd.read_excel(in_path, sheet_name=None, header=None)
        except Exception as e:
            print(f"[fail] {in_path.name}: {e}")
            continue

        per_file_changed = 0
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                df2, stats = _zero_eem_df(
                    df,
                    ex_threshold=args.ex_threshold,
                    em_threshold=args.em_threshold,
                    inclusive=inclusive,
                    ex_axis=args.ex_axis,
                )
                per_file_changed += int(stats.get("changed_cells", 0))
                df2.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        total_changed += per_file_changed
        print(f"[ok] {in_path.name} -> {out_path.name} (changed_cells≈{per_file_changed})")

    print(f"Done. output_dir={output_dir} total_changed_cells≈{total_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

