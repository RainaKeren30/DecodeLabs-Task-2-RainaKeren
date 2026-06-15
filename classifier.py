"""
=============================================================
  Data Classification Using AI — Iris Flower Dataset
  Author : Raina Keren
  Purpose: Internship Submission — Supervised ML Project
=============================================================
"""

# ── Standard library & third-party imports ──────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ── Output directory ─────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1 — Loading the Iris Dataset")
print("="*60)

iris = load_iris()

# Build a tidy DataFrame so we can work with column names
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
df["label"]   = iris.target          # numeric label kept separately

print(f"\n✓ Dataset loaded  →  {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Classes  : {list(iris.target_names)}")
print(f"\nFirst 5 rows:\n{df.head()}")


# ─────────────────────────────────────────────────────────────
# STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2 — Exploratory Data Analysis")
print("="*60)

print(f"\nDataset info:")
print(df.describe().round(2))

print(f"\nMissing values:\n{df.isnull().sum()}")

print(f"\nClass distribution:\n{df['species'].value_counts()}")


# ── EDA Plot 1 : Pair-plot ────────────────────────────────────
pair = sns.pairplot(df.drop("label", axis=1), hue="species",
                    palette="Set2", plot_kws={"alpha": 0.7}, diag_kind="kde")
pair.fig.suptitle("Iris Feature Pair Plot", y=1.02, fontsize=14, fontweight="bold")
pair.savefig(f"{OUTPUT_DIR}/01_pairplot.png", dpi=120, bbox_inches="tight")
plt.close()
print("\n✓ Saved → outputs/01_pairplot.png")


# ── EDA Plot 2 : Feature distributions by class ──────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Feature Distributions by Species", fontsize=14, fontweight="bold")
palette = {"setosa": "#4CAF50", "versicolor": "#2196F3", "virginica": "#FF5722"}

for ax, feature in zip(axes.flatten(), iris.feature_names):
    for species_name, grp in df.groupby("species"):
        grp[feature].plot.kde(ax=ax, label=species_name,
                              color=palette[species_name], linewidth=2)
    ax.set_title(feature.replace(" (cm)", "").title())
    ax.legend(fontsize=8)
    ax.set_xlabel("Value (cm)")

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/02_feature_distributions.png", dpi=120, bbox_inches="tight")
plt.close()
print("✓ Saved → outputs/02_feature_distributions.png")


# ── EDA Plot 3 : Correlation heatmap ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
corr = df[iris.feature_names].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, square=True)
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/03_correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()
print("✓ Saved → outputs/03_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────
# STEP 3 — PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3 — Preprocessing")
print("="*60)

X = df[iris.feature_names].values   # feature matrix  (150 × 4)
y = iris.target                      # target vector   (150,)

# Scale features — zero mean, unit variance
# Logistic Regression and distance-based models benefit from this;
# tree-based models are invariant but we scale for consistency.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n✓ Features scaled with StandardScaler")
print(f"  Mean after scaling : {X_scaled.mean(axis=0).round(6)}")
print(f"  Std  after scaling : {X_scaled.std(axis=0).round(4)}")


# ─────────────────────────────────────────────────────────────
# STEP 4 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 4 — Train / Test Split")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.25,       # 25 % held-out test set
    random_state=42,      # reproducible split
    stratify=y            # preserve class proportions in both splits
)

print(f"\n✓ Split complete")
print(f"  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")


# ─────────────────────────────────────────────────────────────
# STEP 5 — TRAIN MODELS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 5 — Training Classifiers")
print("="*60)

models = {
    "Decision Tree"     : DecisionTreeClassifier(max_depth=4, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Random Forest"     : RandomForestClassifier(n_estimators=100, max_depth=4,
                                                  random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc   = accuracy_score(y_test, y_pred)
    cv    = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    cm    = confusion_matrix(y_test, y_pred)
    report= classification_report(y_test, y_pred, target_names=iris.target_names)

    results[name] = {
        "model"   : model,
        "y_pred"  : y_pred,
        "accuracy": acc,
        "cv_mean" : cv.mean(),
        "cv_std"  : cv.std(),
        "cm"      : cm,
        "report"  : report,
    }

    print(f"\n  ── {name} ──")
    print(f"     Test Accuracy    : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"     5-Fold CV Score  : {cv.mean():.4f} ± {cv.std():.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 6 — EVALUATE & VISUALISE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 6 — Evaluation & Visualisations")
print("="*60)

# ── Plot 4 : Confusion matrices for all 3 models ─────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=res["cm"],
                                  display_labels=iris.target_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=11)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/04_confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.close()
print("\n✓ Saved → outputs/04_confusion_matrices.png")


# ── Plot 5 : Accuracy comparison bar chart ───────────────────
names  = list(results.keys())
accs   = [results[n]["accuracy"]  for n in names]
cv_m   = [results[n]["cv_mean"]   for n in names]
cv_s   = [results[n]["cv_std"]    for n in names]

x  = np.arange(len(names))
w  = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - w/2, accs, w, label="Test Accuracy",
               color="#4C72B0", alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + w/2, cv_m, w, yerr=cv_s, label="CV Mean ± Std",
               color="#DD8452", alpha=0.85, capsize=5, edgecolor="white")

ax.set_ylim(0.85, 1.02)
ax.set_ylabel("Accuracy")
ax.set_title("Model Comparison — Test Accuracy vs Cross-Validation",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()
ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/05_model_comparison.png", dpi=120, bbox_inches="tight")
plt.close()
print("✓ Saved → outputs/05_model_comparison.png")


# ── Plot 6 : Decision Tree visualisation ─────────────────────
fig, ax = plt.subplots(figsize=(18, 7))
plot_tree(
    results["Decision Tree"]["model"],
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True, rounded=True, fontsize=9, ax=ax,
)
ax.set_title("Decision Tree — Learned Structure", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/06_decision_tree.png", dpi=120, bbox_inches="tight")
plt.close()
print("✓ Saved → outputs/06_decision_tree.png")


# ── Plot 7 : Feature importances (Random Forest) ─────────────
rf_model     = results["Random Forest"]["model"]
importances  = rf_model.feature_importances_
sorted_idx   = np.argsort(importances)[::-1]
sorted_feats = [iris.feature_names[i] for i in sorted_idx]

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#E74C3C" if i == sorted_idx[0] else "#5DADE2" for i in sorted_idx]
ax.barh(sorted_feats[::-1], importances[sorted_idx[::-1]],
        color=colors[::-1], edgecolor="white")
ax.set_xlabel("Importance Score")
ax.set_title("Random Forest — Feature Importances", fontsize=13, fontweight="bold")
ax.axvline(x=0, color="black", linewidth=0.5)
for i, v in enumerate(importances[sorted_idx[::-1]]):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/07_feature_importances.png", dpi=120, bbox_inches="tight")
plt.close()
print("✓ Saved → outputs/07_feature_importances.png")


# ── Print classification reports ─────────────────────────────
print("\n── Classification Reports ──")
for name, res in results.items():
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(res["report"])


# ─────────────────────────────────────────────────────────────
# STEP 7 — BEST MODEL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 7 — Summary")
print("="*60)

best_name = max(results, key=lambda n: results[n]["cv_mean"])
best      = results[best_name]

print(f"\n  Best model (by CV score): {best_name}")
print(f"  Test Accuracy : {best['accuracy']*100:.2f} %")
print(f"  CV Mean       : {best['cv_mean']*100:.2f} % ± {best['cv_std']*100:.2f} %")
print(f"\n  All output charts saved to ./{OUTPUT_DIR}/\n")
print("="*60 + "\n")
