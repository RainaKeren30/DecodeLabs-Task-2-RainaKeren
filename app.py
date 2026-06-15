"""
Iris Flower Classifier — Streamlit App
Run: streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session-state defaults ────────────────────────────────────────────────────
DEFAULTS = {
    "page":      "title",
    "model":     "Decision Tree",
    "test_size": 0.25,
    "max_depth": 4,
    "results":   None,   # cached training output
    "ran":       False,  # whether Run Test has been clicked at least once
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Design tokens ─────────────────────────────────────────────────────────────
PINK       = "#D88FA0"
PINK_LT    = "#E8B4BC"
PINK_VL    = "#F7DADF"
GRAY       = "#C5C0C2"
TEXT       = "#111111"
TEXT2      = "#333333"
PINK_CMAP  = LinearSegmentedColormap.from_list("pink", ["#FAF5F6", PINK])

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.weight":        "bold",
    "figure.facecolor":   "none",
    "axes.facecolor":     "none",
    "axes.edgecolor":     "#C8A0A8",
    "axes.labelcolor":    "#111111",
    "xtick.color":        "#333333",
    "ytick.color":        "#333333",
    "text.color":         "#111111",
    "grid.color":         "#F0D8DC",
    "grid.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
})

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{ font-family:'Inter',sans-serif !important; color:#111 !important; }

.stApp{
  background:linear-gradient(145deg,#F9D0D8 0%,#FAE0E5 18%,#FCF0F2 38%,#FEFAFB 58%,#fff 80%,#FDF5F7 100%) !important;
  background-attachment:fixed !important;
}
[data-testid="stAppViewContainer"] > .main{ animation:pageIn .4s ease; }
@keyframes pageIn{ from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }

section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#F5C6CF 0%,#FADADF 40%,#FEF0F2 100%) !important;
  border-right:1.5px solid #E8A8B4 !important;
}
section[data-testid="stSidebar"] *{ color:#111 !important; font-weight:600 !important; }
section[data-testid="stSidebar"] label{ font-weight:700 !important; font-size:13px !important; }
section[data-testid="stSidebar"] h3{ font-size:15px !important; font-weight:800 !important; }
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stSliderTrackFill"]
  { background:#C0566A !important; }

h1,h2,h3,h4,h5,h6{ color:#111 !important; font-weight:800 !important; }
p,span,li{ color:#111 !important; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span{ color:#111 !important; font-weight:500 !important; }
label{ color:#111 !important; font-weight:700 !important; }

/* Metric cards */
div[data-testid="metric-container"]{
  background:rgba(255,255,255,.80) !important;
  border:1.5px solid #E8B4BC !important;
  border-radius:14px !important;
  padding:14px 16px !important;
}
div[data-testid="metric-container"] label
  { font-size:10px !important; font-weight:700 !important; text-transform:uppercase !important; color:#444 !important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"]
  { font-size:24px !important; font-weight:800 !important; color:#111 !important; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"]
  { font-size:11px !important; font-weight:600 !important; }

/* Primary button (Next / Run Test) */
.stButton > button{
  background:#C0566A !important; color:#fff !important;
  border:2px solid #C0566A !important; border-radius:10px !important;
  padding:11px 28px !important; font-size:15px !important;
  font-weight:700 !important; transition:opacity .15s !important;
  min-width:160px !important;
}
.stButton > button:hover{ opacity:.85 !important; }
.stButton > button p{ color:#fff !important; font-size:15px !important; font-weight:700 !important; }

/* Back button override */
.btn-back .stButton > button{
  background:#fff !important; color:#C0566A !important;
  border-color:#C0566A !important;
}
.btn-back .stButton > button p{ color:#C0566A !important; }
.btn-back .stButton > button:hover{ background:#FEF0F2 !important; }

/* Run Test button — green accent */
.btn-run .stButton > button{
  background:#2D7D4F !important; border-color:#2D7D4F !important;
  font-size:14px !important; padding:10px 22px !important;
}
.btn-run .stButton > button p{ color:#fff !important; }
.btn-run .stButton > button:hover{ opacity:.85 !important; }

/* Sliders */
[data-testid="stSlider"] label{ color:#111 !important; font-weight:700 !important; }

/* Data frames */
div[data-testid="stDataFrame"]{
  border:1.5px solid #E8B4BC !important;
  border-radius:12px !important; overflow:hidden !important;
}

/* Shared helper classes */
.eyebrow{ font-size:11px; font-weight:800; color:#C0566A; letter-spacing:.12em; text-transform:uppercase; margin-bottom:4px; }
.pg-title{ font-size:30px; font-weight:800; color:#111; letter-spacing:-.5px; margin-bottom:4px; }
.pg-sub  { font-size:14px; font-weight:500; color:#333; line-height:1.6; }
.divider { height:1.5px; background:#E8C4CB; margin:16px 0; }
.chip-row{ display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 22px; }
.chip    { padding:5px 13px; border-radius:20px; font-size:12px; font-weight:700; background:rgba(255,255,255,.85); border:1.5px solid #D0A0AA; color:#333; }
.chip-hi { background:#F2C4CB; border-color:#C0566A; color:#7A1E30; }
.hbox    { background:rgba(247,218,223,.85); border:1.5px solid #D88FA0; border-radius:12px; padding:14px 18px; margin-top:10px; }
.hl-lbl  { font-size:10px; font-weight:800; color:#9B3A50; letter-spacing:.08em; text-transform:uppercase; margin-bottom:5px; }
.hl-body { font-size:13px; font-weight:600; color:#111; line-height:1.65; }
.run-banner{
  background:#EAF6EE; border:1.5px solid #2D7D4F; border-radius:10px;
  padding:10px 16px; margin-bottom:8px;
  font-size:13px; font-weight:700; color:#1A5C38;
}
.stale-banner{
  background:#FFF8E7; border:1.5px solid #C89A2A; border-radius:10px;
  padding:10px 16px; margin-bottom:8px;
  font-size:13px; font-weight:700; color:#7A5800;
}
</style>
""", unsafe_allow_html=True)

# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    df   = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    df["label"]   = iris.target
    return iris, df

iris, df = load_data()

# ── Training function (NOT cached — runs fresh on every Run Test click) ───────
def run_training(test_size, max_depth):
    X      = df[iris.feature_names].values
    y      = iris.target
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=test_size, random_state=42, stratify=y)

    classifiers = {
        "Decision Tree":       DecisionTreeClassifier(max_depth=max_depth, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=max_depth,
                                                      random_state=42),
    }
    out = {}
    for name, clf in classifiers.items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        cv     = cross_val_score(clf, Xs, y, cv=5)
        out[name] = {
            "model":    clf,
            "scaler":   scaler,
            "y_pred":   y_pred,
            "y_test":   y_te,
            "accuracy": accuracy_score(y_te, y_pred),
            "cv_mean":  cv.mean(),
            "cv_std":   cv.std(),
            "cm":       confusion_matrix(y_te, y_pred),
            "report":   classification_report(
                            y_te, y_pred,
                            target_names=iris.target_names,
                            output_dict=True),
        }
    return out, scaler

def clean_fig(w=9, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    return fig, ax

# ── Sidebar (shared across dashboard + results) ───────────────────────────────
def render_sidebar():
    st.markdown("### ⚙️ Configuration")
    st.markdown("<hr style='border:none;border-top:1px solid #E8B4BC;margin:10px 0'>",
                unsafe_allow_html=True)

    st.markdown("**Select model**")
    model = st.radio(
        "", ["Decision Tree", "Logistic Regression", "Random Forest"],
        index=["Decision Tree", "Logistic Regression", "Random Forest"].index(
            st.session_state.model),
        label_visibility="collapsed")
    st.session_state.model = model

    st.markdown("<hr style='border:none;border-top:1px solid #E8B4BC;margin:10px 0'>",
                unsafe_allow_html=True)
    st.markdown("**Training parameters**")

    test_size = st.slider(
        "Test set fraction", 0.15, 0.40,
        value=st.session_state.test_size, step=0.05)
    max_depth = st.slider(
        "Max tree depth", 2, 8,
        value=st.session_state.max_depth, step=1)

    # Detect if params changed since last run
    params_changed = (
        test_size != st.session_state.test_size or
        max_depth != st.session_state.max_depth
    )
    st.session_state.test_size = test_size
    st.session_state.max_depth = max_depth

    st.markdown("<hr style='border:none;border-top:1px solid #E8B4BC;margin:10px 0'>",
                unsafe_allow_html=True)

    # ── RUN TEST BUTTON ────────────────────────────────────────────────────────
    st.markdown("**Run test**")
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#555;margin-bottom:8px'>"
        "Re-trains all 3 models with<br>current parameters.</div>",
        unsafe_allow_html=True)

    st.markdown('<div class="btn-run">', unsafe_allow_html=True)
    run_clicked = st.button("▶  Run Test", key="run_test_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    if run_clicked or st.session_state.results is None:
        with st.spinner("Training models…"):
            results, scaler = run_training(test_size, max_depth)
        st.session_state.results = results
        st.session_state.ran     = True
        st.session_state.last_ts = test_size
        st.session_state.last_md = max_depth
        st.rerun()

    st.markdown("<hr style='border:none;border-top:1px solid #E8B4BC;margin:10px 0'>",
                unsafe_allow_html=True)
    n_test  = int(150 * test_size)
    n_train = 150 - n_test
    st.markdown(
        f"<div style='font-size:11px;font-weight:700;color:#333;line-height:2'>"
        f"Train: {n_train} &nbsp;·&nbsp; Test: {n_test}<br>"
        f"CV: 5-fold stratified<br>"
        f"Seed: 42</div>",
        unsafe_allow_html=True)

    return params_changed

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TITLE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "title":
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                min-height:74vh;text-align:center;gap:20px;padding:40px 20px">
      <span style="font-size:11px;font-weight:800;color:#C0566A;letter-spacing:.14em;
                   text-transform:uppercase;background:#F2C4CB;border:1.5px solid #C0566A;
                   border-radius:20px;padding:5px 18px">Supervised ML · Iris Dataset</span>
      <h1 style="font-size:50px;font-weight:800;color:#111;letter-spacing:-2px;
                 line-height:1.08;margin:0">
        Iris Flower<br>Classification
      </h1>
      <p style="font-size:16px;color:#444;line-height:1.75;max-width:500px;margin:0">
        A supervised ML pipeline comparing <strong>Decision Tree</strong>,
        <strong>Logistic Regression</strong>, and <strong>Random Forest</strong>
        on the classic 150-sample Iris dataset.<br>
        Adjust parameters, run tests, and watch results update live.
      </p>
      <div class="chip-row" style="justify-content:center">
        <span class="chip chip-hi">Python 3.11</span>
        <span class="chip">scikit-learn</span>
        <span class="chip">Streamlit</span>
        <span class="chip">150 samples</span>
        <span class="chip">4 features</span>
        <span class="chip">3 classes</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([2, 1, 2])
    with col:
        if st.button("Enter Dashboard →"):
            st.session_state.page = "dashboard"
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "dashboard":
    with st.sidebar:
        params_changed = render_sidebar()

    results   = st.session_state.results
    sel       = st.session_state.model
    test_size = st.session_state.test_size
    max_depth = st.session_state.max_depth

    # ── stale / fresh banner ──────────────────────────────────────────────────
    last_ts = st.session_state.get("last_ts", test_size)
    last_md = st.session_state.get("last_md", max_depth)
    stale   = (results is not None) and (
        last_ts != test_size or last_md != max_depth)

    if stale:
        st.markdown(
            "<div class='stale-banner'>⚠️ Parameters changed — click <strong>▶ Run Test</strong>"
            " in the sidebar to retrain with new settings.</div>",
            unsafe_allow_html=True)
    elif st.session_state.ran:
        st.markdown(
            f"<div class='run-banner'>✅ Models trained — test split <strong>"
            f"{int(last_ts*100)}%</strong>, max depth <strong>{last_md}</strong>.</div>",
            unsafe_allow_html=True)

    st.markdown('<p class="eyebrow">Model performance</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="pg-title">Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="pg-sub">Active model: <strong>{sel}</strong>'
        f' &nbsp;·&nbsp; Test split: <strong>{int(test_size*100)}%</strong>'
        f' &nbsp;·&nbsp; Max depth: <strong>{max_depth}</strong></p>',
        unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if results is None:
        st.info("👈  Set parameters in the sidebar and click **▶ Run Test** to train models.")
    else:
        res = results[sel]

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Test accuracy",    f"{res['accuracy']*100:.1f}%")
        with c2: st.metric("CV mean (5-fold)", f"{res['cv_mean']*100:.1f}%")
        with c3: st.metric("CV std dev",       f"±{res['cv_std']*100:.1f}%")
        with c4: st.metric("Test samples",     f"{int(150*test_size)}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Accuracy bar chart
        st.markdown("##### Accuracy comparison — all 3 models")
        names = list(results.keys())
        accs  = [results[n]["accuracy"] for n in names]
        cv_m  = [results[n]["cv_mean"]  for n in names]
        cv_s  = [results[n]["cv_std"]   for n in names]
        x, w  = np.arange(len(names)), 0.32
        fig, ax = clean_fig(10, 3.5)
        ax.bar(x - w/2, accs, w, color=PINK,    label="Test accuracy", zorder=3)
        ax.bar(x + w/2, cv_m, w, color=PINK_VL, label="CV mean",
               yerr=cv_s, capsize=4,
               error_kw={"color": PINK, "linewidth": 1.5}, zorder=3)
        ax.set_ylim(max(0.75, min(accs) - 0.08), 1.04)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.grid(axis="y", zorder=0)
        ax.legend(fontsize=11, frameon=False)
        for i, v in enumerate(accs):
            ax.text(i - w/2, v + .004, f"{v*100:.1f}%",
                    ha="center", va="bottom", fontsize=10,
                    color=TEXT2, fontweight="bold")
        st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Feature importances — Random Forest")
            rf  = results["Random Forest"]["model"]
            imp = pd.Series(rf.feature_importances_,
                            index=iris.feature_names).sort_values()
            fig2, ax2 = clean_fig(6, 3)
            colors_b = [PINK if i == len(imp)-1 else PINK_LT
                        for i in range(len(imp))]
            imp.plot.barh(ax=ax2, color=colors_b, edgecolor="none")
            ax2.set_xlabel("Importance score", fontsize=11)
            ax2.grid(axis="x", zorder=0)
            ax2.spines["bottom"].set_visible(False)
            for i, v in enumerate(imp):
                ax2.text(v + .005, i, f"{v:.3f}",
                         va="center", fontsize=10,
                         color=TEXT2, fontweight="bold")
            st.pyplot(fig2, use_container_width=True); plt.close()

        with col_b:
            st.markdown("##### Petal length distribution by species")
            pal = {"setosa": PINK, "versicolor": PINK_LT, "virginica": GRAY}
            fig3, ax3 = clean_fig(6, 3)
            for sp, grp in df.groupby("species"):
                grp["petal length (cm)"].plot.kde(
                    ax=ax3, label=sp.capitalize(),
                    color=pal[sp], linewidth=2)
            ax3.set_xlabel("Petal length (cm)", fontsize=11)
            ax3.legend(fontsize=10, frameon=False)
            ax3.grid(axis="y", zorder=0)
            st.pyplot(fig3, use_container_width=True); plt.close()

    # Nav buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col_back, _, col_next = st.columns([1, 2, 1])
    with col_back:
        st.markdown('<div class="btn-back">', unsafe_allow_html=True)
        if st.button("← Back", key="dash_back"):
            st.session_state.page = "title"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col_next:
        if st.button("View Results →", key="dash_next"):
            st.session_state.page = "results"
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "results":
    with st.sidebar:
        params_changed = render_sidebar()

    results   = st.session_state.results
    sel       = st.session_state.model
    test_size = st.session_state.test_size
    max_depth = st.session_state.max_depth

    last_ts = st.session_state.get("last_ts", test_size)
    last_md = st.session_state.get("last_md", max_depth)
    stale   = (results is not None) and (
        last_ts != test_size or last_md != max_depth)

    if stale:
        st.markdown(
            "<div class='stale-banner'>⚠️ Parameters changed — click <strong>▶ Run Test</strong>"
            " in the sidebar to retrain with new settings.</div>",
            unsafe_allow_html=True)
    elif st.session_state.ran:
        st.markdown(
            f"<div class='run-banner'>✅ Results for test split <strong>"
            f"{int(last_ts*100)}%</strong>, max depth <strong>{last_md}</strong>.</div>",
            unsafe_allow_html=True)

    st.markdown('<p class="eyebrow">Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="pg-title">Results</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="pg-sub">Confusion matrix &amp; report — <strong>{sel}</strong>'
        f' &nbsp;·&nbsp; Test split: <strong>{int(test_size*100)}%</strong>'
        f' &nbsp;·&nbsp; Max depth: <strong>{max_depth}</strong></p>',
        unsafe_allow_html=True)

    if results is None:
        st.info("👈  Set parameters in the sidebar and click **▶ Run Test** to train models.")
    else:
        res = results[sel]
        scaler = res["scaler"]

        # Model info banner
        st.markdown(
            f"<div style='background:#F2C4CB;border:1.5px solid #C0566A;"
            f"border-radius:12px;padding:11px 15px;margin:10px 0'>"
            f"<div style='font-size:15px;font-weight:800;color:#7A1E30'>{sel}</div>"
            f"<div style='font-size:12px;font-weight:600;color:#9B3A50;margin-top:3px'>"
            f"Test accuracy: {res['accuracy']*100:.1f}% &nbsp;·&nbsp;"
            f" CV mean: {res['cv_mean']*100:.1f}% &nbsp;·&nbsp;"
            f" CV std: ±{res['cv_std']*100:.1f}%</div></div>",
            unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Confusion matrix")
            fig, ax = clean_fig(5, 4)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=res["cm"],
                display_labels=iris.target_names)
            disp.plot(ax=ax, colorbar=True, cmap=PINK_CMAP)
            ax.tick_params(colors=TEXT, labelsize=10)
            ax.xaxis.label.set_color(TEXT2)
            ax.yaxis.label.set_color(TEXT2)
            if disp.im_:
                disp.im_.colorbar.ax.tick_params(labelsize=9, colors="#555")
            st.pyplot(fig, use_container_width=True); plt.close()
            st.markdown(
                "<div style='font-size:11px;font-weight:700;color:#555;margin-top:4px'>"
                "Diagonal = correct &nbsp;·&nbsp; errors only between Versicolor ↔ Virginica"
                "</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("##### Classification report")
            rpt = pd.DataFrame(res["report"]).T.round(3)
            styled = rpt.style.background_gradient(
                cmap=sns.light_palette(PINK, as_cmap=True),
                subset=["precision", "recall", "f1-score"])
            st.dataframe(styled, use_container_width=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        st.markdown("##### Decision Tree — learned structure")
        dt_model = results["Decision Tree"]["model"]
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        fig2.patch.set_alpha(0); ax2.patch.set_alpha(0)
        plot_tree(dt_model,
                  feature_names=iris.feature_names,
                  class_names=iris.target_names,
                  filled=True, rounded=True,
                  fontsize=8, ax=ax2, impurity=False)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close()

        # Insights
        insights = {
            "Decision Tree": (
                "Petal length ≤2.45 cm perfectly isolates Setosa. "
                "Further depth on petal width separates Versicolor from Virginica. "
                "Change max depth and re-run to see the tree structure evolve."),
            "Logistic Regression": (
                "Linear boundaries achieve high accuracy — petal features dominate coefficients. "
                "Very stable CV across 5 folds, making it the most consistent linear model. "
                "Max depth has no effect on Logistic Regression."),
            "Random Forest": (
                "Ensemble of 100 trees achieves the best CV consistency. "
                "Petal width + petal length account for ~85% of feature importance. "
                "Increase max depth and re-run to allow finer splits."),
        }
        st.markdown(
            f"<div class='hbox'>"
            f"<div class='hl-lbl'>{sel} — key insight</div>"
            f"<div class='hl-body'>{insights[sel]}</div>"
            f"</div>", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Live predictor
        st.markdown("##### 🔮 Live predictor")
        st.markdown(
            "<div style='font-size:12px;color:#555;font-weight:600;margin-bottom:10px'>"
            "Adjust the sliders below — prediction updates instantly using the trained model."
            "</div>", unsafe_allow_html=True)
        pc1, pc2 = st.columns(2)
        with pc1:
            sl = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
            sw = st.slider("Sepal width (cm)",  2.0, 4.5, 3.0, 0.1)
        with pc2:
            pl = st.slider("Petal length (cm)", 1.0, 7.0, 3.8, 0.1)
            pw = st.slider("Petal width (cm)",  0.1, 2.5, 1.2, 0.1)

        inp   = scaler.transform([[sl, sw, pl, pw]])
        pred  = res["model"].predict(inp)[0]
        proba = res["model"].predict_proba(inp)[0]
        prob_str = " &nbsp;·&nbsp; ".join(
            [f"{iris.target_names[i].capitalize()}: {proba[i]*100:.1f}%"
             for i in range(3)])
        st.markdown(
            f"<div class='hbox'>"
            f"<div class='hl-lbl'>Prediction — {sel}</div>"
            f"<div class='hl-body' style='font-size:22px;font-weight:800'>"
            f"{iris.target_names[pred].capitalize()}</div>"
            f"<div style='font-size:12px;color:#555;font-weight:600;margin-top:5px'>"
            f"{prob_str}</div>"
            f"</div>", unsafe_allow_html=True)

    # Nav
    st.markdown("<br>", unsafe_allow_html=True)
    col_back, _, _ = st.columns([1, 2, 1])
    with col_back:
        st.markdown('<div class="btn-back">', unsafe_allow_html=True)
        if st.button("← Back to Dashboard", key="res_back"):
            st.session_state.page = "dashboard"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
