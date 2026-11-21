"""
Step 3 — Train the FINAL model and log it to MLflow.

Decision (from src/experiment.py, results/k_selection.csv):
  * K=2 has the highest silhouette (0.290) but it only reproduces the
    Channel split (Horeca vs Retail) — no new business action.
  * K=3: silhouette 0.259, Davies-Bouldin 1.341 (best of K=2..8),
    bootstrap stability ARI mean 0.925 (20 x 80% subsamples),
    seed stability ARI 0.979. Three genuinely different baskets.
  * K>=4: silhouette drops to ~0.19 and stability to 0.65-0.82.
  => Final model: KMeans K=3 on log1p + StandardScaler of the 6 spending
     columns. Channel / Region are NOT inputs — they are used only to
     describe each segment afterwards.

Segments found (means, m.u. / year):
  Segment 0 — "Retail Grocery Staples"   n=80  (18%), total ~29,548
      Grocery-led, low Fresh/Frozen. 60% Retail.
      Action: weekly grocery + detergents bundle, bulk pricing on staples.
  Segment 1 — "High-Value Full Basket"   n=147 (33%), total ~52,903
      Highest on EVERY category (Fresh 17,043, Grocery 13,334, Milk 10,560).
      Action: priority delivery slot, dedicated account manager,
      volume discount / loyalty pricing — protect this segment first.
  Segment 2 — "Horeca Fresh & Frozen"    n=213 (48%), total ~21,028
      99% Horeca. Fresh-led (11,939) + Frozen, very low Milk/Grocery.
      Action: frequent fresh/frozen deliveries, cold-chain priority,
      early-morning delivery window for restaurants/cafes.

Run:
    python src/train.py [--no-mlflow]
Outputs:
    models/kmeans_pipeline.joblib   (log1p + scaler + KMeans, one object)
    models/segment_profiles.json    (names, sizes, means, actions)
    results/segment_profiles.csv
    results/figures/silhouette.png, pca_clusters.png, profile_means.png
    MLflow run in ./mlruns  (experiment "wholesale-segmentation")
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "wholesale_customers.csv"
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
RESULTS = ROOT / "results"; (RESULTS / "figures").mkdir(parents=True, exist_ok=True)

SPEND = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
K = 3
SEGMENTS = {
    0: {"name": "Retail Grocery Staples",
        "action": "Weekly grocery + detergents bundle; bulk pricing on staples; standard delivery route."},
    1: {"name": "High-Value Full Basket",
        "action": "Priority delivery slot + dedicated account manager; volume / loyalty pricing; protect first — highest spend (mean ~52,900 m.u.)."},
    2: {"name": "Horeca Fresh & Frozen",
        "action": "Frequent fresh + frozen deliveries with cold-chain priority; early-morning window for hotels / restaurants / cafes."},
}

def build_pipeline(k=K, seed=42):
    return Pipeline([
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed)),
    ])

def main(use_mlflow=True):
    df = pd.read_csv(DATA)
    X = df[SPEND]
    pipe = build_pipeline()
    labels = pipe.fit_predict(X)
    df["segment"] = labels
    df["segment_name"] = df["segment"].map(lambda i: SEGMENTS[i]["name"])
    Xs = pipe[:-1].transform(X)
    sil = float(silhouette_score(Xs, labels))
    db = float(davies_bouldin_score(Xs, labels))
    print(f"Final KMeans K={K}: silhouette={sil:.4f}, Davies-Bouldin={db:.4f}")
    print(df["segment"].value_counts().sort_index().to_string())

    # --- profiles ---
    prof = df.groupby("segment")[SPEND].mean().round(0)
    prof["n"] = df.groupby("segment").size()
    prof["pct"] = (prof["n"] / len(df) * 100).round(1)
    prof["total_spend_mean"] = df.assign(t=df[SPEND].sum(axis=1)).groupby("segment")["t"].mean().round(0)
    prof["pct_retail"] = df.groupby("segment")["Channel"].apply(lambda s: round(float((s == 2).mean() * 100), 1))
    prof["pct_horeca"] = 100 - prof["pct_retail"]
    prof["name"] = [SEGMENTS[i]["name"] for i in prof.index]
    prof["action"] = [SEGMENTS[i]["action"] for i in prof.index]
    prof.to_csv(RESULTS / "segment_profiles.csv")
    print(prof.to_string())

    profiles_json = {
        "k": K, "silhouette": round(sil, 4), "davies_bouldin": round(db, 4),
        "features": SPEND, "n_customers": int(len(df)),
        "segments": {
            str(i): {"name": SEGMENTS[i]["name"], "action": SEGMENTS[i]["action"],
                     "n": int((labels == i).sum()),
                     "mean_spend": {c: float(prof.loc[i, c]) for c in SPEND},
                     "total_spend_mean": float(prof.loc[i, "total_spend_mean"]),
                     "pct_retail": float(prof.loc[i, "pct_retail"])}
            for i in range(K)
        },
    }
    (MODELS / "segment_profiles.json").write_text(json.dumps(profiles_json, indent=2))
    joblib.dump(pipe, MODELS / "kmeans_pipeline.joblib")
    df.to_csv(RESULTS / "customers_with_segments.csv", index=False)
    print(f"Saved models/kmeans_pipeline.joblib, models/segment_profiles.json")

    # --- figures ---
    # 1. Silhouette plot
    sample_sil = silhouette_samples(Xs, labels)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_lower = 10
    for i in range(K):
        vals = np.sort(sample_sil[labels == i])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * len(vals), f"Segment {i}")
        y_lower = y_upper + 10
    ax.axvline(sil, color="red", linestyle="--", label=f"Mean silhouette = {sil:.3f}")
    ax.set_title("Silhouette plot — KMeans K=3"); ax.set_xlabel("Silhouette coefficient"); ax.set_ylabel("Customer")
    ax.legend(); plt.tight_layout(); plt.savefig(RESULTS / "figures" / "silhouette.png", dpi=150); plt.close()

    # 2. PCA scatter (2 components of the scaled log-spending)
    pca = PCA(n_components=2, random_state=42).fit(Xs)
    pts = pca.transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(K):
        ax.scatter(pts[labels == i, 0], pts[labels == i, 1], s=18, alpha=0.7, label=f"{i} — {SEGMENTS[i]['name']}")
    ax.set_title(f"Customer segments (PCA, {pca.explained_variance_ratio_.sum():.0%} variance)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(RESULTS / "figures" / "pca_clusters.png", dpi=150); plt.close()

    # 3. Mean spend per segment (log scale, so small categories are visible)
    fig, ax = plt.subplots(figsize=(9, 4))
    prof[SPEND].plot(kind="bar", ax=ax, logy=True)
    ax.set_title("Mean annual spend per segment (log scale, m.u.)"); ax.set_ylabel("Mean spend (log scale)")
    ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(RESULTS / "figures" / "profile_means.png", dpi=150); plt.close()

    # --- MLflow ---
    if use_mlflow:
        try:
            import mlflow
            from mlflow.models import infer_signature
            # SQLite backend (not the file store): MLflow >= 3 refuses
            # file-store tracking, and the model registry needs a DB backend.
            mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
            mlflow.set_experiment("wholesale-segmentation")
            with mlflow.start_run(run_name=f"kmeans-k{K}"):
                mlflow.log_param("model", "KMeans")
                mlflow.log_param("k", K)
                mlflow.log_param("features", ",".join(SPEND))
                mlflow.log_param("preprocessing", "log1p + StandardScaler")
                mlflow.log_param("n_init", 10); mlflow.log_param("random_state", 42)
                mlflow.log_metric("silhouette", sil)
                mlflow.log_metric("davies_bouldin", db)
                mlflow.log_metric("n_customers", len(df))
                for i in range(K):
                    mlflow.log_metric(f"segment_{i}_n", int((labels == i).sum()))
                mlflow.log_artifact(str(RESULTS / "segment_profiles.csv"))
                mlflow.log_artifact(str(RESULTS / "k_selection.csv"))
                sig = infer_signature(X.head(5), pipe.predict(X.head(5)))
                mlflow.sklearn.log_model(pipe, name="model", signature=sig,
                                         registered_model_name="wholesale-segmentation")
                print("MLflow run logged. Registered model: wholesale-segmentation (version created, no stage yet).")
                print("Promote it only via the CI/CD workflow — see .github/workflows/ci-cd.yml")
        except Exception as e:
            print(f"MLflow logging skipped ({type(e).__name__}: {e}). Model files above are still saved.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    main(use_mlflow=not a.no_mlflow)
