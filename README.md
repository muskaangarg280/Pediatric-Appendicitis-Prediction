# Pediatric Appendicitis Prediction using Machine Learning

## Overview
This project applies **machine learning techniques** to support the **diagnosis, severity assessment, and treatment management** of pediatric appendicitis using real-world clinical data.  
The work was completed as part of a **university group project** for a Data Mining & Machine Learning module.

The work evaluates and compares **tabular, image-based, and multimodal approaches** to understand their effectiveness in clinical decision support.

---

## Problem Context
Pediatric appendicitis is a common cause of acute abdominal pain in children and is frequently misdiagnosed due to overlapping symptoms and variability in clinical expertise. Misclassification can lead to delayed treatment or unnecessary surgery.

This project aims to develop **data-driven predictive models** that:
- Improve diagnostic accuracy
- Classify disease severity
- Support treatment management decisions

---

## Dataset
The project uses the **Regensburg Pediatric Appendicitis Dataset**, a publicly available and ethically approved dataset containing:
- 500+ pediatric patient records  
- 50+ clinical features (demographics, lab tests, physical exams, diagnostic scores)  
- 1,700+ ultrasound images  
- Target variables:
  - Diagnosis (appendicitis / no appendicitis)
  - Severity (absent, uncomplicated, complicated)
  - Management (conservative vs surgical)

The dataset reflects real-world clinical challenges, including **missing values and class imbalance**, which are addressed through preprocessing and synthetic data generation.

---

## Methods & Models
The project evaluates multiple approaches:

### Tabular Data Models
- Naive Bayes  
- Logistic Regression  
- Random Forest  
- Extra Trees  
- XGBoost  
- Multi-Layer Perceptron (MLP)

### Image-Based Models
- Convolutional Neural Networks (CNNs) trained on ultrasound images

### Multimodal Learning
- Hybrid neural networks combining tabular clinical data and image features

Models were evaluated using **accuracy, F1-score, ROC-AUC, PR-AUC**, and confusion matrices.

---

## My Contributions
My work focused on **tabular data modelling and analysis**, specifically:

- Performed **data preprocessing** on clinical tabular data:
  - Missing value handling
  - Feature encoding
  - Data cleaning and preparation
- Conducted **exploratory data analysis (EDA)** to:
  - Identify key predictive clinical features
  - Analyse feature distributions and correlations
  - Understand class imbalance and its impact on modelling
- Designed and implemented a **Multi-Layer Perceptron (MLP)** for:
  - Diagnosis prediction
  - Severity classification
  - Treatment management prediction
- Evaluated models using:
  - Accuracy
  - F1-score
  - ROC-AUC and PR-AUC
  - Confusion matrices
- Contributed to analytical discussion and interpretation of model results

---

## Key Results (MLP – Tabular Data)
- **Severity prediction:** ~80% accuracy with balanced precision and recall  
- **Management prediction:** ~87% accuracy, F1 ≈ 0.86  
- Demonstrated that **structured clinical features** were more predictive than imaging data in this setting

---

## Overall Findings
- Models trained on **tabular clinical data** consistently outperformed image-only and multimodal models.
- The **MLP trained on tabular data** achieved the strongest overall performance, particularly for severity and management prediction.
- Image-based and multimodal models were limited by data sparsity and variability in ultrasound images.

---

## Methods Used
- Machine Learning: MLP (neural networks)
- Evaluation Metrics: Accuracy, F1-score, ROC-AUC, PR-AUC
- Tools & Libraries:
  - Python
  - Scikit-learn
  - TensorFlow / Keras
  - Pandas, NumPy
  - Matplotlib

---

## Note
This project was completed as part of a **group university assignment**.  
This repository documents **my specific contributions and learning outcomes** and does not imply sole ownership of the full project.

---

