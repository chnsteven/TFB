#!/usr/bin/env python3
"""Aggregate hourly SH event CSVs (TFB long format) to daily series."""

import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
INPUT_DIR = os.path.join(ROOT, "dataset", "forecasting")


def aggregate_to_daily(csv_path: str, output_path: str) -> None:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.floor("D")
    daily = (
        df.groupby(["day", "cols"], as_index=False)["data"]
        .mean()
        .rename(columns={"day": "date"})
    )
    daily.to_csv(output_path, index=False)
    print(
        f"Wrote {output_path}: {daily.shape[0]} rows, "
        f"{daily['cols'].nunique()} variables, {daily['date'].nunique()} days"
    )


def main() -> None:
    for name in sorted(os.listdir(INPUT_DIR)):
        if not name.startswith("event_") or not name.endswith(".csv"):
            continue
        if name.endswith("_daily.csv"):
            continue
        in_path = os.path.join(INPUT_DIR, name)
        out_path = os.path.join(INPUT_DIR, name.replace(".csv", "_daily.csv"))
        aggregate_to_daily(in_path, out_path)


if __name__ == "__main__":
    main()
