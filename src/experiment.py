"""Step 2 — Clustering experiments: K=2..8, silhouette, inertia, stability.
Spending columns only (Channel/Region are kept for business interpretation,
NOT for distance — they are nominal and Region is the UCI target column).
Preprocessing: log1p (spending is very right-skewed) + StandardScaler.
Stability: 20 bootstrap subsamples (80%), Adjusted Rand Index vs full-data labels.
Also: KMeans seed stability (10 seeds) and Agglomerative / GMM comparison at best K.
Saves results/*.csv + results/figures/*.png and prints a summary table.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "wholesale_customers.csv"
OUT = ROOT / "results"
(OUT / "figures").mkdir(parents=True, exist_ok=True)

SPEND = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

def make_pipeline(k, seed=42):
    return Pipeline([
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed)),
    ])

def main():
    df = pd.read_csv(DATA)
    X_raw = df[SPEND]
    print(f"Data: {df.shape}, spend cols: {SPEND}")
    print("Skew before log:\n", X_raw.skew().round(2).to_string())

    rows = []
    for k in range(2, 9):
        pipe = make_pipeline(k)
        labels = pipe.fit_predict(X_raw)
        Xs = pipe[:-1].transform(X_raw)
        sil = silhouette_score(Xs, labels)
        db = davies_bouldin_score(Xs, labels)
        inertia = pipe[-1].inertia_
        # stability: 20 x 80% subsample, ARI on the overlapping points
        aris = []
        rng = np.random.default_rng(42)
        for _ in range(20):
            idx = rng.choice(len(df), size=int(0.8 * len(df)), replace=False)
            sub = make_pipeline(k, seed=int(rng.integers(0, 10000)))
            sub_labels = sub.fit_predict(X_raw.iloc[idx])
            # compare subsample labels to full labels restricted to idx
            aris.append(adjusted_rand_score(labels[idx], sub_labels))
        # seed stability: 10 seeds, ARI vs seed-42 labels
        seed_aris = []
        for s in range(10):
            l2 = make_pipeline(k, seed=s).fit_predict(X_raw)
            seed_aris.append(adjusted_rand_score(labels, l2))
        rows.append(dict(k=k, silhouette=round(float(sil), 4),
                         davies_bouldin=round(float(db), 4),
                         inertia=round(float(inertia), 1),
                         stability_ari_mean=round(float(np.mean(aris)), 4),
                         stability_ari_min=round(float(np.min(aris)), 4),
                         seed_ari_mean=round(float(np.mean(seed_aris)), 4)))
        print(rows[-1])

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "k_selection.csv", index=False)

    # elbow + silhouette plot
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(res.k, res.inertia, marker="o"); ax[0].set_title("Elbow — Inertia vs K"); ax[0].set_xlabel("K"); ax[0].set_ylabel("Inertia")
    ax[1].plot(res.k, res.silhouette, marker="o", color="teal"); ax[1].plot(res.k, res.stability_ari_mean, marker="s", color="orange", label="Stability ARI")
    ax[1].set_title("Silhouette & Stability vs K"); ax[1].set_xlabel("K"); ax[1].legend()
    plt.tight_layout(); plt.savefig(OUT / "figures" / "k_selection.png", dpi=150); plt.close()

    best_k = int(res.loc[res.silhouette.idxmax(), "k"])
    print(f"\nBest K by silhouette: {best_k} (silhouette={res.silhouette.max():.4f})")
    # Also report the K with best silhouette*stability trade-off for the report
    res["combined"] = res.silhouette * res.stability_ari_mean
    print(res.sort_values("combined", ascending=False).head(3).to_string(index=False))

    # Compare algorithms at best K
    Xs_pipe = Pipeline([("log", FunctionTransformer(np.log1p, validate=False)), ("scaler", StandardScaler())])
    Xs = Xs_pipe.fit_transform(X_raw)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xs)
    agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(Xs)
    gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=3).fit(Xs)
    for name, lab in [("KMeans", km.labels_), ("Agglomerative-Ward", agg.labels_), ("GMM", gmm.predict(Xs))]:
        print(f"{name:20s} silhouette={silhouette_score(Xs, lab):.4f}  DB={davies_bouldin_score(Xs, lab):.4f}  sizes={np.bincount(lab)}")
    print(f"KMeans vs Agglomerative ARI: {adjusted_rand_score(km.labels_, agg.labels_):.4f}")
    print(f"KMeans vs GMM ARI: {adjusted_rand_score(km.labels_, gmm.predict(Xs)):.4f}")

if __name__ == "__main__":
    main()
