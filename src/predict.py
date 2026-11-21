"""Shared prediction logic — used by the Streamlit app and by scripts.
Normalises column names (Delicatessen -> Delicassen, case-insensitive),
validates the 6 spending columns, and attaches segment + action.
"""
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEND = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
ALIASES = {"delicatessen": "Delicassen", "delicassen": "Delicassen",
           "detergents_paper": "Detergents_Paper", "detergents paper": "Detergents_Paper"}

def load_model():
    return joblib.load(ROOT / "models" / "kmeans_pipeline.joblib")

def load_profiles():
    return json.loads((ROOT / "models" / "segment_profiles.json").read_text())

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [ALIASES.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
    # title-case the known spend columns if they came in lower case
    rename = {c: next((s for s in SPEND if s.lower() == c.lower()), c) for c in df.columns}
    return df.rename(columns=rename)

def predict(df: pd.DataFrame, model=None, profiles=None):
    """Return df + segment, segment_name, action, total_spend columns."""
    model = model or load_model()
    profiles = profiles or load_profiles()
    df = normalise_columns(df)
    missing = [c for c in SPEND if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. "
                         f"Upload needs: {SPEND}")
    out = df.copy()
    out["segment"] = model.predict(out[SPEND])
    out["segment_name"] = out["segment"].map(lambda i: profiles["segments"][str(i)]["name"])
    out["action_suggestion"] = out["segment"].map(lambda i: profiles["segments"][str(i)]["action"])
    out["total_spend"] = out[SPEND].sum(axis=1)
    return out

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "raw" / "wholesale_customers.csv")
    df = pd.read_csv(path)
    res = predict(df)
    print(res[["segment", "segment_name", "total_spend"]].head(10).to_string())
    print(res["segment_name"].value_counts().to_string())
    res.to_csv(ROOT / "results" / "predictions.csv", index=False)
    print("Saved results/predictions.csv")
