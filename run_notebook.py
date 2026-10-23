"""Validate local inputs and execute a course notebook's plain Python cells."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "lstm": "01_lstm_final.ipynb",
    "lstm-exploration": "02_lstm_exploration.ipynb",
    "keras": "03_lstm_keras_baseline.ipynb",
    "prophet": "04_prophet_baseline.ipynb",
    "ensemble": "05_ensemble_exploration.ipynb",
}


def check_data():
    """Check the input contract without importing the ML dependencies."""
    seen_dates = set()
    for filename in ("train.csv", "val.csv", "test.csv"):
        path = ROOT / "data" / "raw" / filename
        if not path.is_file():
            raise ValueError(f"Missing {path}. See data/README.md.")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            header = handle.readline()
            handle.seek(0)
            reader = csv.DictReader(handle, delimiter="\t" if "\t" in header else ",")
            required = {"Relative Date"}
            if filename != "test.csv":
                required.add("Predict Target")
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{filename}: expected columns {sorted(required)}")
            rows = list(reader)
        minimum = 1 if filename == "test.csv" else 13
        if len(rows) < minimum:
            raise ValueError(f"{filename}: requires at least {minimum} rows")
        dates = set()
        for number, row in enumerate(rows, start=2):
            date = (row.get("Relative Date") or "").strip()
            if not date or date in dates or date in seen_dates:
                raise ValueError(f"{filename}:{number}: missing or repeated date")
            dates.add(date)
            if filename != "test.csv":
                try:
                    target = float(row.get("Predict Target", ""))
                except (ValueError, TypeError):
                    raise ValueError(f"{filename}:{number}: invalid target") from None
                if not math.isfinite(target):
                    raise ValueError(f"{filename}:{number}: target must be finite")
        seen_dates.update(dates)
        print(f"OK {filename}: {len(rows)} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=NOTEBOOKS)
    parser.add_argument("--check-data", action="store_true", help="validate data and exit without training")
    args = parser.parse_args()
    try:
        check_data()
    except ValueError as error:
        parser.exit(2, f"Input error: {error}\n")
    if args.check_data:
        return

    notebook_path = ROOT / "notebooks" / NOTEBOOKS[args.experiment]
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(notebook_path)}
    os.chdir(ROOT)
    print(f"Running {notebook_path.name}; outputs: outputs/{args.experiment}/", flush=True)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            exec(compile(code, f"{notebook_path.name}:cell-{index}", "exec"), namespace)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
