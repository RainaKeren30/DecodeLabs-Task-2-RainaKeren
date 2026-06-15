# VS Code Setup Guide — Step by Step Commands

## Prerequisites

Install these before starting:
- Python 3.10+ → https://www.python.org/downloads/ (check "Add Python to PATH")
- VS Code → https://code.visualstudio.com/
- Git (optional) → https://git-scm.com/

---

## Setup Commands

Open VS Code, press Ctrl+` to open the terminal, then run each command.

### Step 1 — Navigate to the project folder

```
cd path\to\iris_classifier
```

Example on Windows:
```
cd C:\Users\Raina\Desktop\iris_classifier
```

Example on Mac/Linux:
```
cd ~/Desktop/iris_classifier
```

---

### Step 2 — Create a virtual environment

```
python -m venv venv
```

---

### Step 3 — Activate the virtual environment

Windows:
```
venv\Scripts\activate
```

Mac / Linux:
```
source venv/bin/activate
```

You will see `(venv)` appear at the start of the terminal prompt. This means it worked.

---

### Step 4 — Install all dependencies

```
pip install -r requirements.txt
```

Wait for all packages to finish (about 1–3 minutes). You will see a success message at the end.

---

### Step 5 — Run the core ML script

```
python classifier.py
```

This will:
- Load and explore the Iris dataset
- Train all 3 classifiers
- Print accuracy, confusion matrix, classification report
- Save 7 chart images to the outputs/ folder

---

### Step 6 — Run the Streamlit multi-page app

```
streamlit run app.py
```

A browser window opens automatically at:
```
http://localhost:8501
```

The app has 3 pages:
- Title page — project overview and stats
- Dashboard — model metrics, accuracy chart, feature importances
- Results — confusion matrix, classification report, live predictor

Use the sidebar to navigate between pages.

Press Ctrl+C in the terminal to stop the Streamlit server.

---

## Selecting the Python Interpreter in VS Code

1. Press Ctrl+Shift+P
2. Type: Python: Select Interpreter
3. Select the one showing ./venv/... or .\venv\Scripts\python.exe

---

## Project Folder After Setup

```
iris_classifier/
├── venv/                    created by Step 2
├── outputs/                 created after Step 5
│   ├── 01_pairplot.png
│   ├── 02_feature_distributions.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_confusion_matrices.png
│   ├── 05_model_comparison.png
│   ├── 06_decision_tree.png
│   └── 07_feature_importances.png
├── classifier.py
├── app.py
├── requirements.txt
├── README.md
├── project_explanation.md
└── vscode_setup_guide.md
```

---

## Common Errors and Fixes

| Error | Fix |
|---|---|
| `python` not found | Re-install Python and check "Add Python to PATH" |
| `(venv)` not showing | Run the activate command again from Step 3 |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `streamlit: command not found` | Make sure venv is activated, then retry |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Charts not appearing | Check the outputs/ folder was created |

---

## One-Line Quick Start (after first setup)

Next time you open VS Code, just run these three commands:

Windows:
```
venv\Scripts\activate
python classifier.py
streamlit run app.py
```

Mac/Linux:
```
source venv/bin/activate
python classifier.py
streamlit run app.py
```
