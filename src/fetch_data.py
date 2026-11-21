"""
Step 1 — Fetch the UCI Wholesale Customers dataset (id=292) exactly
as shown in the project brief / second screenshot.

    pip install ucimlrepo
    from ucimlrepo import fetch_ucirepo
    wholesale_customers = fetch_ucirepo(id=292)

Run:
    python src/fetch_data.py
Saves:
    data/raw/wholesale_customers.csv   (440 rows)
    data/raw/wholesale_customers_metadata.txt
"""
from pathlib import Path
import pandas as pd

try:
    from ucimlrepo import fetch_ucirepo
except ImportError as e:
    raise SystemExit("ucimlrepo not installed. Run: pip install -r requirements.txt") from e

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def main():
    wholesale_customers = fetch_ucirepo(id=292)

    X = wholesale_customers.data.features
    y = wholesale_customers.data.targets
    print("Features shape:", X.shape)
    print("Targets shape:", None if y is None else y.shape)
    print("\nFeatures columns:", list(X.columns))
    print("\nMetadata:\n", wholesale_customers.metadata)
    print("\nVariables:\n", wholesale_customers.variables)

    df = X.copy()
    if y is not None and len(y.columns):
        # y is empty for this dataset in ucimlrepo; keep the guard anyway
        for c in y.columns:
            if c not in df.columns:
                df[c] = y[c]

    out = RAW / "wholesale_customers.csv"
    df.to_csv(out, index=False)
    (RAW / "wholesale_customers_metadata.txt").write_text(
        f"{wholesale_customers.metadata}\n\n{wholesale_customers.variables}\n"
    )
    print(f"\nSaved {len(df)} rows x {len(df.columns)} cols -> {out}")
    print(df.head().to_string())
    print("\nMissing values:\n", df.isna().sum().to_string())
    print("\nDescribe:\n", df.describe().to_string())

if __name__ == "__main__":
    main()
