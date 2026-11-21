"""
Distributor Customer Segmentation — Streamlit app, v2 (P37, Ali Abbas)

Rebuilt to match `Segmentation · Professional Screens.html`:
  Dashboard  |  Customers  |  Score  |  Model
  Inter font, #f6f7f9 background, white cards, accent #2f5bea,
  segment colours: 0 purple #7c5cd6, 1 orange #e08a1e, 2 teal #12a594.

Corrections to the HTML mock (checked against the trained model):
  * Total annual spend is 14.6M m.u., not 14.2M (sum of all 6 spend
    columns over 440 customers = 14,619,500).
  * Model chip says "Staging", not "Production" — MLflow registry has
    wholesale-segmentation v1 in Staging; Production needs manual
    approval (see .github/workflows/promote.yml).
  * Single-customer scoring uses the REAL trained pipeline
    (log1p + StandardScaler + KMeans, models/kmeans_pipeline.joblib),
    not the HTML mock's nearest-segment-mean shortcut.
  * PCA is computed live from the scaled data (PC1 44%, PC2 27%),
    not from hard-coded coordinates.

Run:  streamlit run app/streamlit_app.py   ->  http://localhost:8501
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predict import SPEND, load_model, load_profiles, predict  # noqa: E402

SEG_COLOURS = ["#7c5cd6", "#e08a1e", "#12a594"]
ACCENT = "#2f5bea"
REGION_NAMES = {1: "Lisbon", 2: "Oporto", 3: "Other"}
CHANNEL_NAMES = {1: "Horeca", 2: "Retail"}
PAGE_SIZE = 12

st.set_page_config(page_title="Segmentation · Wholesale Customers",
                   page_icon="▦", layout="wide")

CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: Inter, system-ui, sans-serif; }
.stApp { background: #f6f7f9; color: #1c2430; }
header[data-testid="stHeader"] { display: none; }
.block-container { padding-top: 0 !important; max-width: 1240px; }
.box { background: #fff; border: 1px solid #e4e7ec; border-radius: 8px; }
.kp { display: grid; grid-template-columns: repeat(4, 1fr); }
.kp > div { padding: 14px 18px; border-right: 1px solid #e4e7ec; }
.kp > div:last-child { border: 0; }
.kp small { color: #667085; display: block; }
.kp b { font-size: 22px; font-weight: 600; }
.kp em { font-style: normal; color: #667085; font-size: 12px; margin-left: 6px; }
.bh { padding: 12px 16px; border-bottom: 1px solid #e4e7ec; display: flex;
      justify-content: space-between; align-items: center; }
.bh b { font-weight: 600; }
.bc { color: #667085; font-size: 12px; }
table.prof { border-collapse: collapse; width: 100%; font-size: 13px; }
table.prof th { font-weight: 500; color: #667085; text-align: left;
                padding: 8px 14px; border-bottom: 1px solid #e4e7ec; background: #f6f7f9; }
table.prof td { padding: 9px 14px; border-bottom: 1px solid #e4e7ec; }
table.prof tr:last-child td { border: 0; }
.r { text-align: right; font-variant-numeric: tabular-nums; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 7px; }
.mb { height: 6px; width: 90px; background: #e4e7ec; border-radius: 3px;
      display: inline-block; vertical-align: middle; margin-right: 8px; overflow: hidden; }
.mb i { display: block; height: 100%; }
.kv div { display: flex; justify-content: space-between; padding: 7px 0;
          border-bottom: 1px solid #e4e7ec; gap: 16px; }
.kv div:last-child { border: 0; }
.kv span { color: #667085; }
.chip { display: inline-block; padding: 1px 8px; border: 1px solid #e4e7ec;
        border-radius: 4px; font-size: 12px; background: #fff; }
.out { padding: 12px 14px; border: 1px solid #e4e7ec; border-radius: 6px;
       background: #fff; }
[data-testid="stDataFrame"] { border: 1px solid #e4e7ec; border-radius: 8px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- data
@st.cache_resource
def _model():
    return load_model(), load_profiles()

@st.cache_data
def _customers():
    path = ROOT / "results" / "customers_with_segments.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:  # fallback: score the raw file
        df = predict(pd.read_csv(ROOT / "data" / "raw" / "wholesale_customers.csv"))
    df = df.copy()
    df.insert(0, "ID", range(1, len(df) + 1))
    df["Channel_name"] = df["Channel"].map(CHANNEL_NAMES)
    df["Region_name"] = df["Region"].map(REGION_NAMES)
    df["total_spend"] = df[SPEND].sum(axis=1)
    return df

@st.cache_data
def _k_table():
    return pd.read_csv(ROOT / "results" / "k_selection.csv")

@st.cache_data
def _pca_points():
    df = _customers()
    model, _ = _model()
    Xs = model[:-1].transform(df[SPEND])
    pca = PCA(n_components=2, random_state=42).fit(Xs)
    pts = pca.transform(Xs)
    return pts, pca.explained_variance_ratio_, df["segment"].to_numpy()

try:
    model, profiles = _model()
    customers = _customers()
    k_table = _k_table()
except Exception as e:
    st.error(f"Model / results not found. Train first: `python src/train.py`  ({e})")
    st.stop()

SEG_NAMES = [profiles["segments"][str(i)]["name"] for i in range(3)]
SEG_ACTIONS = [profiles["segments"][str(i)]["action"] for i in range(3)]
SEG_N = [int(profiles["segments"][str(i)]["n"]) for i in range(3)]
TOTAL_SPEND = float(customers["total_spend"].sum())

def seg_tag(i):
    return (f'<span class="dot" style="background:{SEG_COLOURS[i]}"></span>'
            f"{SEG_NAMES[i]}")

def fmt(n):
    return f"{int(round(n)):,}"

# ---------------------------------------------------------------- header
if "page" not in st.session_state:
    st.session_state.page = "dashboard"
if "q" not in st.session_state:
    st.session_state.q = ""

def go(page):
    st.session_state.page = page

h1, h2, h3 = st.columns([2.2, 5.2, 1.1])
with h1:
    st.markdown('<div style="font-weight:600;font-size:14px;padding-top:8px">'
                '<span style="display:inline-block;width:20px;height:20px;border-radius:5px;'
                f'background:{ACCENT};vertical-align:-4px;margin-right:8px"></span>'
                "Segmentation</div>", unsafe_allow_html=True)
with h2:
    nav = st.columns(4)
    for j, (key, label) in enumerate([("dashboard", "Dashboard"), ("customers", "Customers"),
                                      ("score", "Score"), ("model", "Model")]):
        with nav[j]:
            st.button(label, key=f"nav_{key}",
                      type="primary" if st.session_state.page == key else "secondary",
                      use_container_width=True, on_click=go, args=(key,))
with h3:
    st.markdown('<div style="text-align:right;padding-top:6px"><span style="display:inline-grid;place-items:center;'
                'width:28px;height:28px;border-radius:50%;background:#f2f4f7;border:1px solid #e4e7ec;'
                'font-size:11px;font-weight:600">AA</span></div>', unsafe_allow_html=True)

search = st.text_input("Search customers", placeholder="Search customers",
                       label_visibility="collapsed", key="global_search")
if search and search != st.session_state.q:
    st.session_state.q = search
    st.session_state.page = "customers"
st.divider()

# ---------------------------------------------------------------- charts
def pca_fig():
    pts, var, labels = _pca_points()
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for i in range(3):
        ax.scatter(pts[labels == i, 0], pts[labels == i, 1], s=22,
                   c=SEG_COLOURS[i], alpha=0.62, label=SEG_NAMES[i])
    ax.set_xlabel(f"PC1 ({var[0]:.0%})"); ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    ax.grid(color="#e4e7ec", linewidth=0.6); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), ncol=3)
    plt.tight_layout()
    return fig

def category_fig():
    prof = pd.read_csv(ROOT / "results" / "segment_profiles.csv").set_index("segment")
    cats = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
    labels = ["Fresh", "Milk", "Grocery", "Frozen", "Det. paper", "Delicassen"]
    mx = prof[cats].to_numpy().max()
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for ci, cat in enumerate(cats):
        y = len(cats) - ci
        for k in range(3):
            ax.barh(y - 0.28 + k * 0.28, prof.loc[k, cat] / mx,
                    height=0.22, color=SEG_COLOURS[k])
        ax.text(0, y + 0.42, labels[ci], fontsize=9, color="#667085", va="bottom")
    ax.set_xlim(0, 1.05); ax.set_ylim(0.3, len(cats) + 0.7)
    ax.set_yticks([]); ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout()
    return fig

def k_fig():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.axvspan(2.5, 3.5, color=ACCENT, alpha=0.08)
    ax.plot(k_table["k"], k_table["silhouette"], marker="o", color=ACCENT, label="Silhouette")
    ax.plot(k_table["k"], k_table["stability_ari_mean"], marker="o",
            color=SEG_COLOURS[1], label="Stability ARI")
    ax.set_xticks(k_table["k"]); ax.set_xlabel("K")
    ax.grid(color="#e4e7ec", linewidth=0.6); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------- Dashboard
if st.session_state.page == "dashboard":
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<div class="bc">Workspace / Dashboard</div>'
                    '<h1 style="margin:2px 0 0;font-size:18px">Customer segments</h1>',
                    unsafe_allow_html=True)
    with c2:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Export", use_container_width=True):
                st.session_state.page = "customers"
                st.rerun()
        with b2:
            st.button("Score new file", type="primary", use_container_width=True,
                      on_click=go, args=("score",))
    st.markdown(
        f'<div class="box kp"><div><small>Customers</small><b>{len(customers)}</b></div>'
        f'<div><small>Total annual spend</small><b>{TOTAL_SPEND / 1e6:.1f}M</b><em>m.u.</em></div>'
        f'<div><small>Silhouette</small><b>{profiles["silhouette"]:.3f}</b></div>'
        f'<div><small>Stability (ARI)</small><b>0.925</b></div></div>',
        unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.5, 1])
    with left:
        st.markdown('<div class="box"><div class="bh"><b>Segment map</b>'
                    '<span class="bc">PCA of log-scaled spend · PC1 44% · PC2 27%</span></div>',
                    unsafe_allow_html=True)
        st.pyplot(pca_fig(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="box"><div class="bh"><b>Average spend by category</b></div>',
                    unsafe_allow_html=True)
        st.pyplot(category_fig(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    prof = pd.read_csv(ROOT / "results" / "segment_profiles.csv")
    rows = ""
    for _, r in prof.iterrows():
        k = int(r["segment"])
        rows += (f"<tr><td>{seg_tag(k)}</td><td class='r'>{int(r['n'])}</td>"
                 f"<td><span class='mb'><i style='width:{r['pct'] / 50 * 100:.0f}%;"
                 f"background:{SEG_COLOURS[k]}'></i></span>{r['pct']:.1f}%</td>"
                 f"<td class='r'>{fmt(r['total_spend_mean'])}</td>"
                 f"<td class='r'>{r['pct_retail']:.0f}%</td><td>{SEG_ACTIONS[k]}</td></tr>")
    st.markdown('<div class="box"><div class="bh"><b>Segments</b></div>'
                '<table class="prof"><thead><tr><th>Segment</th><th class="r">Customers</th>'
                '<th>Share</th><th class="r">Avg spend</th><th class="r">Retail</th>'
                f'<th>Recommended action</th></tr></thead><tbody>{rows}</tbody></table></div>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------- Customers
elif st.session_state.page == "customers":
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<div class="bc">Workspace / Customers</div>'
                    '<h1 style="margin:2px 0 0;font-size:18px">Customers</h1>',
                    unsafe_allow_html=True)
    with c2:
        st.download_button("Download CSV", customers.to_csv(index=False).encode(),
                           "customers_with_segments.csv", "text/csv", use_container_width=True)
    if "cust_page" not in st.session_state:
        st.session_state.cust_page = 0
    f1, f2, f3, f4 = st.columns([2, 2, 1.4, 1.6])
    with f1:
        q = st.text_input("Filter by ID", value=st.session_state.q,
                          placeholder="Filter by ID", label_visibility="collapsed")
    with f2:
        seg_filter = st.selectbox("Segment", ["All segments"] + SEG_NAMES,
                                  label_visibility="collapsed")
    with f3:
        ch_filter = st.selectbox("Channel", ["All channels", "Horeca", "Retail"],
                                 label_visibility="collapsed")
    with f4:
        sort_by = st.selectbox("Sort by", ["Total ↓", "Total ↑", "ID", "Fresh", "Milk",
                                           "Grocery", "Frozen", "Det. paper", "Delicassen"],
                               label_visibility="collapsed")
    df = customers.copy()
    if q:
        df = df[df["ID"].astype(str).str.startswith(str(q).strip())]
    if seg_filter != "All segments":
        df = df[df["segment_name"] == seg_filter]
    if ch_filter != "All channels":
        df = df[df["Channel_name"] == ch_filter]
    sort_map = {"Total ↓": ("total_spend", False), "Total ↑": ("total_spend", True),
                "ID": ("ID", True), "Fresh": ("Fresh", False), "Milk": ("Milk", False),
                "Grocery": ("Grocery", False), "Frozen": ("Frozen", False),
                "Det. paper": ("Detergents_Paper", False), "Delicassen": ("Delicassen", False)}
    col, asc = sort_map[sort_by]
    df = df.sort_values(col, ascending=asc)
    n_pages = max(1, (len(df) + PAGE_SIZE - 1) // PAGE_SIZE)
    st.session_state.cust_page = min(st.session_state.cust_page, n_pages - 1)
    page_df = df.iloc[st.session_state.cust_page * PAGE_SIZE:
                      (st.session_state.cust_page + 1) * PAGE_SIZE]
    show = pd.DataFrame({
        "ID": page_df["ID"].map(lambda i: f"#{i:03d}"),
        "Channel": page_df["Channel_name"], "Region": page_df["Region_name"],
        "Fresh": page_df["Fresh"], "Milk": page_df["Milk"], "Grocery": page_df["Grocery"],
        "Frozen": page_df["Frozen"], "Det. paper": page_df["Detergents_Paper"],
        "Delicassen": page_df["Delicassen"], "Total": page_df["total_spend"],
        "Segment": page_df["segment_name"],
    })
    st.markdown(f'<div class="box"><div class="bh"><span></span>'
                f'<span class="bc">{len(df)} customers</span></div>', unsafe_allow_html=True)
    st.dataframe(show, use_container_width=True, hide_index=True)
    p1, p2, p3 = st.columns([2, 2, 1])
    with p1:
        lo = st.session_state.cust_page * PAGE_SIZE + 1 if len(df) else 0
        hi = min(len(df), (st.session_state.cust_page + 1) * PAGE_SIZE)
        st.markdown(f'<span class="bc">{lo}–{hi} of {len(df)}</span>', unsafe_allow_html=True)
    with p2:
        pass
    with p3:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Previous", disabled=st.session_state.cust_page == 0,
                         use_container_width=True):
                st.session_state.cust_page -= 1; st.rerun()
        with b2:
            if st.button("Next", disabled=st.session_state.cust_page >= n_pages - 1,
                         use_container_width=True):
                st.session_state.cust_page += 1; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- Score
elif st.session_state.page == "score":
    st.markdown('<div class="bc">Workspace / Score</div>'
                '<h1 style="margin:2px 0 0;font-size:18px">Score customers</h1>',
                unsafe_allow_html=True)
    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="box"><div class="bh"><b>Upload a file</b></div>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop a CSV or Excel file — up to 10,000 rows",
                                    type=["csv", "xlsx"], label_visibility="visible")
        st.markdown('<div class="kv" style="padding:6px 16px 14px">'
                    '<div><span>Required</span>Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen</div>'
                    '<div><span>Optional</span>Channel, Region, customer name</div>'
                    '<div><span>Output</span>Segment, action, total spend</div></div>',
                    unsafe_allow_html=True)
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
                if len(raw) > 10000:
                    st.error("File has more than 10,000 rows — please split it and upload in parts.")
                else:
                    result = predict(raw, model, profiles)
                    st.success(f"Scored {len(result)} customers.")
                    st.dataframe(result, use_container_width=True)
                    st.bar_chart(result["segment_name"].value_counts())
                    st.download_button("Download results as CSV",
                                       result.to_csv(index=False).encode(),
                                       "segmented_customers.csv", "text/csv")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Could not read / score this file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="box"><div class="bh"><b>Single customer</b></div>',
                    unsafe_allow_html=True)
        vals = {}
        grid = st.columns(2)
        for j, col_name in enumerate(SPEND):
            with grid[j % 2]:
                vals[col_name] = st.number_input(col_name.replace("_", " "),
                                                 min_value=0, value=5000, step=100,
                                                 key=f"single_{col_name}")
        if st.button("Get segment", type="primary"):
            one = predict(pd.DataFrame([vals]), model, profiles).iloc[0]
            k = int(one["segment"])
            st.markdown(
                f'<div class="out" style="border-left:3px solid {SEG_COLOURS[k]}">'
                f"<b>{seg_tag(k)}</b>"
                f"<div style='color:#667085;margin:2px 0 6px'>Total {fmt(one['total_spend'])} m.u.</div>"
                f"{one['action_suggestion']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- Model
elif st.session_state.page == "model":
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<div class="bc">Workspace / Model</div>'
                    '<h1 style="margin:2px 0 0;font-size:18px">Model quality</h1>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<span class="chip">KMeans · K=3 · Staging</span>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.5, 1])
    with left:
        st.markdown('<div class="box"><div class="bh"><b>Silhouette and stability by K</b></div>',
                    unsafe_allow_html=True)
        st.pyplot(k_fig(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="box"><div class="bh"><b>K selection</b></div>', unsafe_allow_html=True)
        rows = ""
        for _, r in k_table.iterrows():
            hl = ' style="background:#f2f4f7;font-weight:600"' if int(r["k"]) == 3 else ""
            rows += (f"<tr{hl}><td>{int(r['k'])}</td><td class='r'>{r['silhouette']:.3f}</td>"
                     f"<td class='r'>{r['davies_bouldin']:.3f}</td>"
                     f"<td class='r'>{r['stability_ari_mean']:.3f}</td></tr>")
        st.markdown('<table class="prof"><thead><tr><th>K</th><th class="r">Silhouette</th>'
                    '<th class="r">Davies-Bouldin</th><th class="r">ARI</th></tr></thead>'
                    f"<tbody>{rows}</tbody></table></div>", unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="box"><div class="bh"><b>Training data</b></div>'
                '<div class="kv" style="padding:6px 16px 14px">'
                '<div><span>Source</span>UCI Wholesale Customers</div>'
                '<div><span>Rows</span>440, no missing values</div>'
                '<div><span>Features</span>6 spend columns, log1p + standard scaling</div>'
                '<div><span>Why K=3</span>Best Davies-Bouldin; three actionable baskets where '
                'K=2 only split Horeca from Retail</div></div></div>',
                unsafe_allow_html=True)
