# cardiometabolic-risk-prediction-ANN
End-to-end cardiometabolic risk classification using TensorFlow/Keras, Sequential and Functional ANNs, feature preprocessing, regularization, and threshold optimization.
# Cardiometabolic Risk Prediction using Artificial Neural Networks

An end-to-end deep learning project that explores cardiometabolic risk classification using **TensorFlow and Keras**, with a focus on data quality, feature engineering, neural network architectures, model evaluation, and reproducible inference.

> **Disclaimer:** This project uses synthetic data for educational and research purposes. The models are not clinically validated and must not be used for medical diagnosis or treatment decisions.
> ## Author

**Madhu Kunchala**

**GitHub:** [Madhukunchala99](https://github.com/Madhukunchala99)

## Project Overview

This project implements and compares multiple Artificial Neural Network (ANN) architectures to classify elevated cardiometabolic risk from demographic, clinical, and lifestyle features.

The objective is to develop a reproducible machine learning workflow while understanding the practical differences between the **Keras Sequential and Functional APIs**.

## Dataset

- **Dataset:** Synthetic Cardiometabolic Neural Network Dataset
- **Records:** 8,000
- **Input Features:** 13
- **Target:** `elevated_cardiometabolic_risk`
- **Problem Type:** Binary classification

**Features include:**
- Age, sex and BMI
- Systolic and diastolic blood pressure
- Total cholesterol and fasting glucose
- Resting heart rate
- Smoking status and family history
- Physical activity, sleep duration and stress level

The dataset is imbalanced, with approximately 10.44% of records belonging to the elevated-risk class.

## Technology Stack

- **Programming:** Python
- **Deep Learning:** TensorFlow, Keras
- **Machine Learning:** Scikit-learn
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Keras, Joblib, JSON
- **Development Environment:** VS Code, Jupyter Notebook

## Methodology

### 1. Data Understanding and EDA

Performed dataset inspection, quality checks, missing-value analysis, duplicate detection, outlier identification, feature distribution analysis, target association analysis, and correlation assessment.

### 2. Data Preprocessing

- Stratified train/validation/test split: **70% / 15% / 15%**
- StandardScaler for numerical and ordinal features
- OneHotEncoder for physical activity categories
- Training-only preprocessing fit to reduce data leakage
- Float32 conversion for TensorFlow compatibility
- Class weighting to address target imbalance

### 3. Neural Network Architectures

Implemented and compared four ANN models:

| Model | Architecture |
|---|---|
| Baseline Sequential ANN | Dense(32) → Dense(16) → Sigmoid |
| Regularized Sequential ANN | Dense + L2 + Dropout |
| Single-Input Functional ANN | Functional API equivalent to baseline |
| Multi-Input Functional ANN | Separate clinical and other-feature branches |

Models were trained using Adam optimization, binary cross-entropy loss, class weighting, and early stopping.

### 4. Model Evaluation

Evaluated the models using accuracy, precision, recall, F1-score, Average Precision, ROC-AUC, confusion matrices, and precision–recall curves.

A classification threshold was selected using validation data with a project-defined minimum recall objective of 80%.

## Validation Model Comparison

| Model | F1-score | Average Precision | ROC-AUC |
|---|---:|---:|---:|
| Baseline Sequential | 0.3603 | 0.3443 | 0.7885 |
| Regularized Sequential | **0.3629** | **0.3595** | **0.8032** |
| Single-Input Functional | 0.3603 | 0.3443 | 0.7885 |
| Multi-Input Functional | 0.3616 | 0.3495 | 0.7996 |

The **Regularized Sequential ANN** was selected based primarily on its validation Average Precision.

## Final Model Performance

**Selected Model:** Regularized Sequential ANN  
**Classification Threshold:** 0.4592  
**Evaluation:** Untouched test dataset (1,200 records)

| Metric | Test Result |
|---|---:|
| Accuracy | 68.58% |
| Precision | 22.25% |
| Recall | **80.80%** |
| F1-score | 0.3489 |
| Average Precision | 0.3513 |
| ROC-AUC | 0.7943 |

### Test Confusion Matrix

| Actual / Predicted | Lower Risk | Elevated Risk |
|---|---:|---:|
| Lower Risk | 722 | 353 |
| Elevated Risk | 24 | 101 |

The final model detected **101 out of 125 elevated-risk records**, meeting the project's recall objective on the test split. However, the substantial number of false positives indicates that precision remains an important limitation.

## Model Artifacts

The project saves three artifacts in the `models/` directory:

| File | Purpose |
|---|---|
| `regularized_sequential_ann.keras` | Trained neural network |
| `preprocessor.joblib` | Fitted feature transformations |
| `model_metadata.json` | Feature names, threshold and model metadata |

The saved model and preprocessor were successfully reloaded and verified to reproduce the original prediction probabilities on sample validation records.

## Key Learnings

- Designed and trained ANNs using both Sequential and Functional APIs.
- Understood regularization, dropout, class weighting and early stopping.
- Implemented multi-input neural network architectures.
- Learned how class imbalance affects evaluation.
- Compared threshold-dependent and threshold-independent metrics.
- Selected a classification threshold using validation data.
- Saved and restored trained neural networks for reproducible inference.

## Limitations

- The dataset is synthetic and not clinically validated.
- The target-generation formula is undocumented.
- Precision is relatively low, resulting in many false positives.
- Model comparisons use a single split and training run.
- The system is intended for educational experimentation, not clinical deployment.

## Future Enhancements

- Compare ANN performance with conventional machine learning baselines.
- Explore probability calibration and additional evaluation strategies.
- Experiment with alternative architectures and hyperparameters.
- Add a Streamlit demonstration interface.
- Develop a FastAPI inference endpoint.
- Add automated tests and improve reproducibility.

## Project Status

**Phase 1 — Completed:** EDA, preprocessing, ANN architectures, training, validation, model comparison, threshold selection, final testing, and model persistence.

**Future phases:** Additional experiments and application development.

---

**Author:** Madhu Kunchala 
**GitHub:** [Madhukunchala99](https://github.com/Madhukunchala99)




