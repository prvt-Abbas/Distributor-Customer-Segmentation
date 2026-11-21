"""
Move a registered MLflow model version between stages.
Used by CI (train.yml -> Staging) and by the manual promotion
workflow (promote.yml -> Production, after GitHub approval).

Usage:
    python scripts/promote.py --to Staging --latest
    python scripts/promote.py --to Production --version 1

With a real MLflow tracking server set MLFLOW_TRACKING_URI to its URL;
locally it defaults to the sqlite DB created by src/train.py.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True, choices=["Staging", "Production", "Archived"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--version", type=int)
    g.add_argument("--latest", action="store_true")
    a = ap.parse_args()

    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    client = MlflowClient()
    name = "wholesale-segmentation"

    if a.latest:
        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            raise SystemExit(f"No versions of '{name}' found. Run: python src/train.py")
        version = max(int(v.version) for v in versions)
    else:
        version = a.version

    client.transition_model_version_stage(name, str(version), a.to, archive_existing_versions=(a.to == "Production"))
    print(f"Model '{name}' version {version} -> {a.to}")
    if a.to == "Production":
        print("This version is now the Production model used by the app.")

if __name__ == "__main__":
    main()
