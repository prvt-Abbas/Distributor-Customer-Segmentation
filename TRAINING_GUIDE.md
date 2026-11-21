# Training guide — Distributor Customer Segmentation, step by step

For a beginner. Read this top to bottom once, then run the commands in
order. Every step says *what* it does, *why*, and *what good output
looks like* — so you can tell a real success from a silent failure.

## 0. What kind of problem is this?

**Clustering (unsupervised learning).** There is no "correct segment"
column to predict — nobody labelled these 440 customers. The model has
to *discover* groups of customers who spend in similar ways, so the
wholesaler can treat each group differently (delivery frequency, pricing).

That is why there is no accuracy score here. Instead we judge a
clustering by:
- **Silhouette score** (−1 to +1, higher is better): is each customer
  closer to its own cluster than to the next cluster? ~0.26 is modest
  but normal for real spending data — baskets overlap.
- **Davies-Bouldin** (lower is better): are clusters compact and far apart?
- **Stability**: if you remove 20% of customers at random and re-cluster,
  do the same customers stay together? Measured with Adjusted Rand Index
  (ARI, 1 = identical). Our K=3 scores 0.925 — very stable.

## 1. Get the dataset — the exact code from your brief

The second screenshot is the official UCI way to load this dataset.
`ucimlrepo` downloads it from archive.ics.uci.edu, dataset **292**.

```python
pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
wholesale_customers = fetch_ucirepo(id=292)
X = wholesale_customers.data.features   # 440 rows x 7 columns
y = wholesale_customers.data.targets    # Region — 1 column
```

In this project that code lives in `src/fetch_data.py`. Run it:

```bash
python src/fetch_data.py
```

Good output: `Features shape: (440, 7)`, `Saved 440 rows x 8 cols`.
What you get: `Channel`, `Fresh`, `Milk`, `Grocery`, `Frozen`,
`Detergents_Paper`, `Delicassen`, `Region`. No missing values.
`Delicassen` is UCI's own spelling — keep it, don't "fix" it in the
data, or your columns won't match the model later.

**Trap to avoid:** `Region` is listed as the *target* by UCI, but this
is a clustering project, not classification. Do **not** train on Region,
and do not use it as an input — it would leak the answer and the
segments would just be regions. We only use it afterwards to check
whether segments differ by region (they don't much: 69–77% "Other"
everywhere).

## 2. Look at the data before modelling

Spending is extremely skewed: a few customers spend 112,151 on Fresh
while the median spends 8,504; Delicatessen skew is 11.15. If you feed
raw money into KMeans, the few huge spenders dominate everything.

Fix, in this order:
1. **`log1p`** — `log(1 + x)`. Shrinks huge values, keeps small ones
   meaningful. A customer spending 10× more is now only ~2.3 log-units
   away, not 90,000 m.u. away.
2. **`StandardScaler`** — subtract mean, divide by standard deviation,
   per column. Otherwise Fresh (mean 12,000) would count far more than
   Delicatessen (mean 1,525) just because its numbers are bigger.

Both steps are inside one scikit-learn `Pipeline`, so the *same*
transform is applied at training time and later in the Streamlit app —
you cannot forget one of them when scoring new customers.

## 3. Try K = 2..8 and pick K — don't guess it

```bash
python src/experiment.py
```

It trains KMeans for each K and reports silhouette, Davies-Bouldin,
inertia (elbow), bootstrap stability (20 × 80% subsamples, ARI) and
seed stability (10 different random seeds). Results land in
`results/k_selection.csv`.

What happened on the real data:

| K | Silhouette | Stability ARI |
|---|---:|---:|
| 2 | 0.290 | 0.963 |
| **3** | **0.259** | **0.925** |
| 4 | 0.189 | 0.822 |
| 5+ | ~0.19 | 0.65–0.73 |

K=2 is best on silhouette — but look at what it *is*: cluster 0 is
99% Horeca, cluster 1 is 71% Retail. It just re-discovers the Channel
column you already had. K=3 loses only 0.03 silhouette, has the best
Davies-Bouldin (1.341), stays stable (0.925), and splits the Retail
side into two different baskets. K≥4 falls off a cliff on both
silhouette and stability. **So K=3 is the final model, and you can
defend it in a viva with numbers, not taste.** Say the K=2 fact out
loud — hiding it looks worse than explaining it.

## 4. Train the final model

```bash
python src/train.py
```

It fits `log1p → StandardScaler → KMeans(n_clusters=3, n_init=10,
random_state=42)` on the 6 spending columns and saves two files:
- `models/kmeans_pipeline.joblib` — the whole pipeline in one object.
  Load it and call `.predict()`; it does the log + scaling itself.
- `models/segment_profiles.json` — segment names, sizes, mean spend,
  and the action for each segment (the app reads this file).

It also logs everything to MLflow (SQLite `mlflow.db`, experiment
`wholesale-segmentation`) and registers the model as
`wholesale-segmentation` v1. Check it with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://localhost:5000
```

Good output: `Final KMeans K=3: silhouette=0.2594, Davies-Bouldin=1.3406`
and `MLflow run logged. Registered model: wholesale-segmentation`.

## 5. Read the segments like a business person, not a statistician

Open `results/segment_profiles.csv` (or the table in the README) and
for each segment ask: *what do they buy a lot of, what do they barely
buy, and are they Horeca or Retail?*

- **Segment 2 (213 customers, 48%) — Horeca Fresh & Frozen.** Fresh
  11,939, Frozen 3,266, Milk/Grocery/Detergents tiny, 99% Horeca.
  They are restaurants/cafes. They need fresh food *often*, early in
  the morning, kept cold. Pricing: don't push grocery bundles at them.
- **Segment 0 (80, 18%) — Retail Grocery Staples.** Grocery 12,570,
  Detergents 5,554, Fresh only 2,899. Small shops selling staples.
  Weekly bundle, bulk price on grocery + detergents.
- **Segment 1 (147, 33%) — High-Value Full Basket.** Highest on
  everything, mean total 52,903 — more than double Segment 2. Protect
  them: priority delivery, account manager, loyalty pricing. Losing
  one of these costs as much as losing two or three Segment 2 customers.

## 6. Score new customers

Command line:

```bash
python src/predict.py data/sample_customers.csv
```

Streamlit app (the deliverable in the brief):

```bash
streamlit run app/streamlit_app.py   # http://localhost:8501
```

Upload a CSV/Excel with the 6 spending columns, or enter one customer
by hand. The app returns segment + action per row and a downloadable
CSV. It uses `src/predict.py`, so the app and the command line can
never disagree.

Try the sample first: `data/sample_customers.csv` has 6 customers,
2 from each segment.

## 7. MLflow registry + CI/CD — what the brief actually requires

- **Registry:** the model is registered, not just a `.pkl` file on
  disk. A registered model has versions, stages, and a history of
  which run produced it.
- **Staging:** every push to `main` runs `.github/workflows/train.yml`
  — fetch, experiment, train, quality gate (silhouette ≥ 0.20), then
  `scripts/promote.py --to Staging --latest`. A bad model fails the
  gate and never reaches Staging.
- **Production — manual approval:** `.github/workflows/promote.yml`
  runs only when a human starts it, and it runs in the GitHub
  `production` environment. You must create that environment once:
  *Settings → Environments → `production` → Required reviewers → add
  yourself / your teacher.* After that, the workflow pauses and waits
  for Approve. No approval, no Production — that is the whole point.
  The script behind it is `scripts/promote.py --to Production
  --version <n>`, followed by a smoke test on the sample file.

## 8. Viva questions you should be able to answer

1. *Why log-transform?* Spending is right-skewed (skew up to 11.15);
   raw money lets a few huge customers dominate the distance.
2. *Why not use Channel/Region as inputs?* They are nominal / the UCI
   target; using them would just reproduce known labels instead of
   discovering baskets.
3. *Why K=3 when K=2 has a better silhouette?* K=2 only splits
   Horeca/Retail; K=3 keeps 0.259 silhouette, best Davies-Bouldin,
   0.925 stability, and three actionable baskets.
4. *What does silhouette 0.259 mean?* Clusters are real but overlapping
   — use them for prioritisation, not as hard customer "types".
5. *How does a new customer get a segment?* Same pipeline:
   log1p → scaler → nearest KMeans centroid; the app does it through
   `src/predict.py`.
6. *How does a model reach Production?* Only via the promote workflow
   after a human approves it in the `production` environment; CI can
   only reach Staging.
