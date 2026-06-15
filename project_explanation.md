# Project Explanation — Internship Submission

## Data Classification Using AI
**Author:** Raina Keren  
**Type:** Supervised Machine Learning · Classification

---

## 1. Problem Statement

Given four physical measurements of an Iris flower (sepal length, sepal width, petal length, petal width), predict which of three species it belongs to: *Setosa*, *Versicolor*, or *Virginica*.

This is a classic **multi-class classification** problem.

---

## 2. Dataset

**Name:** Iris Flower Dataset (R.A. Fisher, 1936)  
**Source:** Built into `sklearn.datasets`  
**Size:** 150 samples, 4 features, 3 balanced classes (50 per class)  
**Why chosen:** Clean, no missing values, well-studied, perfect for demonstrating ML fundamentals.

---

## 3. ML Pipeline — Step by Step

### Step 1 — Load Data
Used `sklearn.datasets.load_iris()`. Converted to a pandas DataFrame for readability.

### Step 2 — Exploratory Data Analysis (EDA)
- Pair plot to visualise inter-feature relationships
- KDE distributions to understand feature spread per class
- Correlation heatmap to detect multicollinearity

### Step 3 — Preprocessing
- Applied `StandardScaler` (zero mean, unit variance). This is essential for Logistic Regression and beneficial for consistency across all models.
- No missing values, so imputation was not needed.

### Step 4 — Train / Test Split
- 75 % training, 25 % testing
- `stratify=y` ensures each split has the same class proportions

### Step 5 — Model Training
Three classifiers trained and compared:
1. **Decision Tree** — rule-based, fully interpretable
2. **Logistic Regression** — linear probabilistic model
3. **Random Forest** — ensemble of 100 decision trees

### Step 6 — Evaluation
- `accuracy_score` for overall performance
- `confusion_matrix` to see per-class errors
- `classification_report` for precision, recall, F1-score
- `cross_val_score` (5-fold) for robust generalisation estimate

---

## 4. Key Results

All three models achieved **96–100 % accuracy** on this dataset, which reflects how well-separated the three Iris classes are. The main confusion occurs between *Versicolor* and *Virginica*, which overlap in feature space.

The **Random Forest** typically achieves the highest cross-validation score due to variance reduction through ensemble averaging.

---

## 5. Visualisations Produced

| File | What it shows |
|---|---|
| `01_pairplot.png` | All feature pairs coloured by species |
| `02_feature_distributions.png` | KDE of each feature per class |
| `03_correlation_heatmap.png` | Pearson correlation between features |
| `04_confusion_matrices.png` | Prediction errors for all 3 models |
| `05_model_comparison.png` | Test accuracy vs CV score bar chart |
| `06_decision_tree.png` | Visual of learned decision rules |
| `07_feature_importances.png` | Which features matter most (RF) |

---

## 6. Sample Output Descriptions

**Pair Plot** — Setosa is visually separated from the other two species across all feature combinations, confirming it is the easiest class to classify. Versicolor and Virginica overlap in sepal space but are separable in petal space.

**Feature Importances (Random Forest)** — Petal length and petal width contribute ~85 % of the total importance, confirming that petal measurements are the most discriminative features.

**Confusion Matrix** — The off-diagonal elements appear only between Versicolor and Virginica, never involving Setosa, which is always perfectly classified.

---

## 7. What I Learned

- How to design a clean ML pipeline from raw data to evaluation
- Why feature scaling matters for gradient-based and distance-based models
- How ensemble methods (Random Forest) reduce overfitting compared to a single Decision Tree
- How to read and interpret precision, recall, and F1-score
- How to build an interactive ML dashboard with Streamlit

---

## 8. Interview Questions & Answers

**Q: What is the difference between precision and recall?**  
A: Precision = of all predicted positives, how many were actually positive. Recall = of all actual positives, how many did we correctly find. Precision matters when false positives are costly (e.g. spam filter). Recall matters when false negatives are costly (e.g. medical diagnosis).

**Q: Why did you use StandardScaler?**  
A: Logistic Regression uses gradient descent; features on different scales cause the optimiser to converge slowly or find a poor solution. Scaling brings all features to comparable ranges without changing information content.

**Q: What is overfitting and how did you address it?**  
A: Overfitting is when a model memorises training data and performs poorly on new data. I used max_depth to limit tree complexity and 5-fold cross-validation to measure generalisation performance.

**Q: What is cross-validation and why is it better than a single train-test split?**  
A: Cross-validation averages performance across multiple folds, giving a lower-variance estimate of generalisation. A single split result can be lucky or unlucky depending on which samples ended up in which set.

**Q: Why did Random Forest outperform a single Decision Tree?**  
A: Random Forest averages predictions from 100 trees, each trained on a random subset of data and features. This reduces variance (overfitting) while keeping bias low — the core idea behind ensemble methods (bagging).

**Q: What would you do differently with a larger, noisier dataset?**  
A: I would add missing-value imputation, outlier detection, feature engineering, hyperparameter tuning with GridSearchCV, and experiment with gradient boosting (XGBoost/LightGBM) for better accuracy.

---

## 9. How to Run Locally (VS Code)

See the detailed VS Code setup guide in `vscode_setup_guide.md`.

---

*This project was built from scratch as part of an AI/ML internship submission.*
