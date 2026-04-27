# Fraud Detection System (ML-Based)

## Overview
This project implements a machine learning-based fraud detection system designed to simulate financial transaction monitoring. It combines supervised learning models with anomaly detection to identify suspicious transactions.

## Features
- Synthetic dataset generation for transaction data
- Feature engineering based on transaction behavior
- Supervised models:
  - Logistic Regression
  - Random Forest
- Unsupervised anomaly detection using Isolation Forest
- Model evaluation using accuracy, precision, recall, and confusion matrix
- Feature importance analysis for interpretability

## Engineered Features
- **Amount-to-average ratio**: Detects abnormal transaction spikes
- **Deviation**: Measures difference from typical transaction behavior
- **Large transaction flag**: Identifies unusually high-value transactions
- **Frequency flag**: Captures high transaction activity
- **Risk score proxy**: Combined behavioral risk indicator

## Results
- Logistic Regression achieved ~96% accuracy with balanced precision/recall
- Random Forest achieved near-perfect accuracy on synthetic data
- Feature importance highlighted behavioral indicators (ratio, deviation) as key predictors
- Isolation Forest successfully flagged anomalous transactions

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn

## Notes
- Dataset is synthetic and rule-based, used for prototyping and experimentation
- Real-world deployment would require:
  - Noisy, real transaction data
  - Class imbalance handling
  - Model validation on unseen distributions