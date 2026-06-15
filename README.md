# 🌸 Data Classification Using AI
### Supervised Machine Learning — Iris Flower Dataset
> **Internship Submission Project** · Raina Keren

---

## 📌 Project Overview

This project demonstrates a complete, end-to-end **supervised classification pipeline** using the classic Iris dataset. It compares three classification algorithms — Decision Tree, Logistic Regression, and Random Forest — and includes both a standalone script and an interactive Streamlit dashboard.

---

## 🗂 Folder Structure

```
iris_classifier/
├── classifier.py          ← Core ML pipeline (script)
├── app.py                 ← Interactive Streamlit dashboard
├── requirements.txt       ← Python dependencies
├── README.md              ← This file
├── outputs/               ← Auto-generated charts (after running)
│   ├── 01_pairplot.png
│   ├── 02_feature_distributions.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_confusion_matrices.png
│   ├── 05_model_comparison.png
│   ├── 06_decision_tree.png
│   └── 07_feature_importances.png
```

---

## ⚙️ Setup & Run

### 1. Clone / download the project

```bash
# If from GitHub
git clone https://github.com/your-username/iris_classifier.git
cd iris_classifier
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4a. Run the core script (terminal output + saved charts)

```bash
python classifier.py
```

Charts are saved to the `outputs/` folder automatically.

### 4b. Run the interactive Streamlit app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## 🧪 Models Used

| Model | Strength |
|---|---|
| Decision Tree | Interpretable, shows splitting logic visually |
| Logistic Regression | Fast, probabilistic, great baseline |
| Random Forest | Ensemble, high accuracy, feature importance |

---

## 📊 Evaluation Metrics

- **Accuracy Score** — % of correct predictions
- **Confusion Matrix** — true vs predicted class breakdown  
- **Classification Report** — precision, recall, F1-score per class
- **5-Fold Cross-Validation** — generalisation check

---

## 💡 Optional Improvements

- Add **SVM** or **KNN** as additional classifiers  
- Try the **Titanic** or **Breast Cancer** datasets  
- Export the trained model with `joblib` for deployment  
- Add **SHAP** explainability plots  
- Deploy the Streamlit app on **Streamlit Community Cloud** (free)

---

## 🧑‍💻 Author

**Raina Keren** — AI/ML Internship Project
