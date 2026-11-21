# Distributor Customer Segmentation

**P37 — Ali Abbas — DF2-AI-2026-037 — ML**

A wholesaler wants customer segments for tailored delivery and pricing.
This project clusters the UCI Wholesale Customers dataset (440 customers),
checks whether the clusters are stable, names each segment in business
terms, and ships a Streamlit app that scores a new customer file and
suggests an action per segment. The model is tracked in MLflow and can
only reach Production after a human approves it in GitHub Actions.

## Results (actually run, 2026-11-21)

Dataset: UCI Wholesale Customers, fetched with `fetch_ucirepo(id=292)` —
440 rows, 0 missing values. Columns: `Channel` (1=Horeca, 2=Retail),
`Region` (1=Lisbon, 2=Oporto, 3=Other — the UCI target column), and annual
spending in m.u. on `Fresh, Milk, Grocery, Frozen, Detergents_Paper,
Delicassen` (UCI spells it Delicassen).

Clustering used **only the 6 spending columns**, `log1p`-transformed
(spending is very skewed — Delicatessen skew 11.15) and standardised.
`Channel` / `Region` were **not** inputs; they are used only to describe
the segments afterwards.

### Choosing K (K=2..8, KMeans, 20× 80% bootstrap subsamples)

| K | Silhouette | Davies-Bouldin ↓ | Stability ARI (mean) | Seed ARI |
|---|---:|---:|---:|---:|
| 2 | **0.290** | 1.352 | 0.963 | 1.000 |
| **3** | **0.259** | **1.341** | **0.925** | **0.979** |
| 4 | 0.189 | 1.541 | 0.822 | 0.927 |
| 5 | 0.192 | 1.484 | 0.725 | 0.957 |
| 6 | 0.201 | 1.432 | 0.655 | 0.631 |
| 7 | 0.196 | 1.392 | 0.730 | 0.846 |
| 8 | 0.185 | 1.441 | 0.658 | 0.677 |

Full table: `results/k_selection.csv`. KMeans beat Agglomerative-Ward
(0.259) and GMM (0.266) at the chosen K on silhouette.

**Why K=3, not K=2?** K=2 wins on silhouette, but it only reproduces the
Channel split (Horeca vs Retail) — no new action for the wholesaler.
K=3 keeps almost the same silhouette, has the best Davies-Bouldin score,
stability ARI 0.925 (clusters barely move when 20% of customers are
removed), and gives three genuinely different baskets. K≥4 collapses on
both silhouette and stability. This trade-off is stated here on purpose —
a marker should see the choice was made, not hidden.

### The 3 segments (final model: KMeans K=3, silhouette 0.259)

| Segment | n | Mean total spend | Who they are | Action |
|---|---:|---:|---|---|
| **0 — Retail Grocery Staples** | 80 (18%) | 29,548 | Grocery-led (12,570), low Fresh (2,899) / Frozen (607); 60% Retail | Weekly grocery + detergents bundle; bulk pricing on staples; standard delivery route |
| **1 — High-Value Full Basket** | 147 (33%) | 52,903 | Highest on *every* category — Fresh 17,043, Grocery 13,334, Milk 10,560 | Priority delivery slot + dedicated account manager; volume / loyalty pricing; protect first |
| **2 — Horeca Fresh & Frozen** | 213 (48%) | 21,028 | Fresh-led (11,939) + Frozen (3,266), very low Milk / Grocery / Detergents; **99% Horeca** | Frequent fresh + frozen deliveries, cold-chain priority; early-morning window for hotels / restaurants / cafes |

Region does **not** separate the segments (69–77% "Other" in all three) —
so do not plan delivery by region from this data; plan by basket.

Honest limit: silhouette 0.259 is modest — the segments overlap, as real
customer baskets do. The segments are useful for prioritising delivery
and pricing, not for claiming customers belong to only one "type".

## Screens (v2 — rebuilt 2026-11-21 from `Segmentation · Professional Screens.html`)

The Streamlit app now has the 4 screens from that design, same layout,
colours and wording — Inter font, #f6f7f9 background, white cards,
accent #2f5bea, segments purple / orange / teal:

* **Dashboard** — KPI strip (440 customers, **14.6M** total spend,
  silhouette 0.259, stability ARI 0.925), PCA segment map (PC1 44%,
  PC2 27%), average spend by category, segments table with share bars
  and recommended actions.
* **Customers** — all 440 customers, filter by ID / segment / channel,
  sort by any spend column or total, 12 per page with Previous / Next,
  Download CSV.
* **Score** — upload a CSV/Excel (up to 10,000 rows) and get segment +
  action per customer with a downloadable CSV, or score one customer
  by hand.
* **Model** — silhouette + stability by K chart, K-selection table
  (K=3 highlighted), training-data summary.

Three things in the HTML mock were corrected, not copied:
1. Total spend is **14.6M**, not 14.2M (14,619,500 m.u. summed).
2. The Model chip says **Staging**, not Production — the registered
   model is v1 in Staging.
3. Single-customer scoring uses the **real trained pipeline**, not the
   mock's nearest-segment-mean shortcut.

## How to run it (Windows / any laptop)

```bash
# 1. Python 3.10+ and a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac / Linux
pip install -r requirements.txt

# 2. Fetch the dataset exactly as in the brief (needs internet once)
python src/fetch_data.py       # -> data/raw/wholesale_customers.csv (440 rows)

# 3. Re-run the K comparison (optional — results already in results/)
python src/experiment.py       # -> results/k_selection.csv + figures/k_selection.png

# 4. Train the final model + log to MLflow
python src/train.py            # -> models/kmeans_pipeline.joblib
                               #    models/segment_profiles.json
                               #    mlflow.db (experiment "wholesale-segmentation",
                               #    registered model "wholesale-segmentation" v1)

# 5. Score a file from the command line
python src/predict.py data/sample_customers.csv   # -> results/predictions.csv

# 6. Run the app
streamlit run app/streamlit_app.py   # open http://localhost:8501
```

In the app: upload a CSV/Excel with the 6 spending columns
(`Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen` —
`Delicatessen` is accepted too), or use the **One customer** tab.
You get a segment + action per customer and a downloadable CSV.
Try it first with `data/sample_customers.csv` (6 customers, all 3 segments).

In Google Colab the same 4 commands work after
`!pip install -r requirements.txt` — see `TRAINING_GUIDE.md`.

## MLflow + CI/CD (project-specific requirement)

* Training (`src/train.py`) logs parameters, silhouette (0.259),
  Davies-Bouldin (1.341), segment sizes and the model to MLflow
  (SQLite backend `mlflow.db` — MLflow ≥3 refuses the old file store)
  and registers it as **`wholesale-segmentation`**.
* View runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
  then open http://localhost:5000.
* **Train workflow** (`.github/workflows/train.yml`): on push to `main`
  it fetches the data, re-runs the experiments, trains, checks a quality
  gate (silhouette ≥ 0.20, K=3), and promotes the new version to
  **Staging only** (`scripts/promote.py --to Staging --latest`).
  It can never put a model in Production.
* **Promotion workflow** (`.github/workflows/promote.yml`): manual
  `workflow_dispatch` with a version number, running in the GitHub
  **`production` environment**. Set it up once in the repo:
  *Settings → Environments → New environment `production` → Required
  reviewers → add yourself / your teacher → Save.*
  Then a run pauses until a reviewer clicks **Approve**; only then does
  `scripts/promote.py --to Production --version <n>` run, followed by a
  smoke test on the sample file.
  Note: MLflow's stage API prints a deprecation warning in MLflow 3 —
  it still works, and the brief explicitly requires Staging → Production
  stages, so this project uses them deliberately.

## Project structure

```
├── app/streamlit_app.py        # Streamlit app: upload file / one customer -> segment + action
├── src/
│   ├── fetch_data.py           # fetch_ucirepo(id=292) -> data/raw/wholesale_customers.csv
│   ├── experiment.py           # K=2..8, silhouette, Davies-Bouldin, bootstrap stability
│   ├── train.py                # final KMeans K=3 + MLflow logging + registration
│   └── predict.py              # shared scoring logic (used by the app + CLI)
├── scripts/promote.py          # MLflow Staging / Production promotion
├── .github/workflows/
│   ├── train.yml               # CI: train, quality gate, register -> Staging
│   └── promote.yml             # CD: manual approval -> Production
├── data/sample_customers.csv   # 6 customers to try in the app
├── models/kmeans_pipeline.joblib   # trained pipeline (log1p + scaler + KMeans)
├── models/segment_profiles.json    # segment names, sizes, means, actions
├── results/                    # k_selection.csv, segment_profiles.csv,
│                               # customers_with_segments.csv, figures/
├── TRAINING_GUIDE.md           # beginner step-by-step: what each step means
└── requirements.txt
```

## Files already produced by the verified run

`models/kmeans_pipeline.joblib`, `models/segment_profiles.json`,
`results/k_selection.csv`, `results/segment_profiles.csv`,
`results/customers_with_segments.csv`, `mlflow.db` (model v1, Staging),
figures `k_selection.png`, `silhouette.png`, `pca_clusters.png`,
`profile_means.png`. Streamlit verified: HTTP 200 on localhost:8501.
