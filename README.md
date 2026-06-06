# Pediatric Appendicitis Prediction — Tabular MLP Pipeline

A clinical machine learning pipeline for predicting paediatric appendicitis outcomes from structured patient data. The pipeline addresses three sequential prediction tasks — **diagnosis**, **severity**, and **treatment management** — using a multi-layer perceptron trained on tabular clinical and laboratory features from the Regensburg Paediatric Appendicitis Dataset.

---

## The Problem

Paediatric appendicitis diagnosis relies on clinical examination, lab tests, and imaging, but misdiagnosis remains common due to overlapping symptoms and variability in clinical experience. Accurate, objective tools for diagnosis, severity stratification, and treatment planning could meaningfully support clinical decision-making.

This project builds a data-driven, three-stage prediction pipeline directly from routinely collected patient data.

---

## Results

| Task | Accuracy | Macro F1 | ROC-AUC | PR-AUC |
|------|----------|----------|---------|--------|
| Diagnosis | 0.70 | ~0.70 | 0.819 | 0.893 |
| Severity | 0.80 | ~0.80 | 0.875 | 0.832 |
| Management | **0.87** | **0.86** | **0.950** | **0.976** |

The tabular MLP outperformed both an image-based CNN (77% accuracy) and a multimodal model (69% accuracy) evaluated on the same dataset, indicating that structured clinical features carry more discriminative signal than ultrasound image features for this task and dataset size.

---

## Dataset

**Regensburg Paediatric Appendicitis Dataset** — 500+ anonymised paediatric cases (Children's Hospital St. Hedwig, Germany, 2016–2021), combining clinical, laboratory, and diagnostic scoring features with three clinically relevant outcomes: diagnosis, severity, and management.

- Source: Marcinkevičs et al. (2024), *Medical Image Analysis* — [add Kaggle/Zenodo link]
- Licence: **CC BY-NC 4.0** (non-commercial, attribution required)

Only the tabular portion of the dataset is used in this repository.

---

## Pipeline Design

The prediction pipeline is structured hierarchically to reflect real clinical decision flow:

1. **Diagnosis** — binary classification (appendicitis vs no appendicitis) on all patients
2. **Severity** — binary classification (complicated vs uncomplicated) on confirmed appendicitis cases only
3. **Management** — binary classification (primary surgical vs conservative) on confirmed appendicitis cases only

**Methodology highlights:**
- Single hierarchical hold-out test split, with appended and non-appendicitis cases handled separately to preserve class balance
- Imputation and scaling fit on training data only — no data leakage into validation or test
- Class weighting to handle imbalance across all three targets
- Diagnosis decision threshold tuned on the validation set via macro-F1 sweep (τ ∈ [0.70, 0.78])
- Evaluated with accuracy, macro F1, ROC-AUC, PR-AUC, and confusion matrices

**MLP architecture:**
```
Input → Dense(128, ReLU) → BatchNorm → Dropout
      → Dense(64, ReLU)  → BatchNorm → Dropout
      → Dense(1, sigmoid)
```
Trained with Adam (lr=1e-3), binary cross-entropy, EarlyStopping and ReduceLROnPlateau callbacks.

---

## Repository Structure

```
Pediatric-Appendicitis-Prediction/
├── raw_data/
│   └── tabular_data.csv              # raw clinical data (source: Kaggle/Zenodo)
├── processed_data/
│   ├── DataProcessing.py             # preprocessing pipeline
│   ├── processed_data.csv            # full cleaned dataset
│   ├── dataset_tabular.csv           # tabular-only view (MLP and EDA input)
│   └── eda/
│       ├── data_analysis_tabular.py  # exploratory analysis
│       └── dataset_tabular_eda.csv   # EDA output
├── mlp_model_tabular_data/
│   ├── mlp.py                        # MLP implementation
│   └── mlp_files/                    # trained models, preprocessors, outputs
└── README.md
```

---

## How to Run

```bash
# 1. Preprocess the raw tabular data
python processed_data/DataProcessing.py

# 2. Exploratory data analysis (optional)
python processed_data/eda/data_analysis_tabular.py

# 3. Train and evaluate the MLP
python mlp_model_tabular_data/mlp.py
```

> Scripts use `Path(__file__)` for path resolution and can be run from the repo root or their own directories.

**Dependencies:**
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn joblib
```

---

## Exploratory Data Analysis

The EDA examines data quality, feature distributions, and relationships with each target variable, including:

- Correlation heatmap across all encoded features
- Top predictors per target (diagnosis, severity, management)
- Distribution histograms across clinical variables
- Alvarado Score vs diagnosis outcome
- WBC count vs management strategy, faceted by severity

Key findings: peritonitis and presumptive diagnosis are the strongest correlates of the diagnosis target; WBC count and Alvarado Score are the most predictive for management decisions.

---

## Tech Stack

Python · TensorFlow/Keras · scikit-learn · pandas · NumPy · Matplotlib · seaborn · joblib

---

## Acknowledgements

Dataset: Marcinkevičs et al. (2024), *Medical Image Analysis*, licensed CC BY-NC 4.0.  
Academic project, Heriot-Watt University Dubai.