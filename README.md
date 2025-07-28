Health Insurance Claim Prediction
This project presents a real-world machine learning model designed for smart triage of health insurance claims in a national healthcare system.

The model predicts whether a claim file requires manual expert review to determine potential deductions or not.

Prediction = 0 → The claim likely has no deductions → No need for manual assessment
Prediction = 1 → The claim may have deductions → Send to assessor for review
This approach helps automate the screening of claim files, reduce expert workload, and speed up the processing pipeline.

⚠️ Note: Due to data privacy regulations, the original dataset is not included. However, the full preprocessing pipeline, model structure, and profiling summary are available.

Project Overview
• Goal: Binary classification — Predict claim approval

• Technique: XGBoost classifier

• Data: 900,000 structured records, 30 features

• Result: 87% accuracy on test data

Repository Contents

• notebooks/: Jupyter notebooks for preprocessing and modeling

• profile_report/: Data profiling summary (HTML or PDF)

• src/: Python scripts

• README.md: Project overview

Key Steps
• Data Cleaning & Preprocessing

• Outlier removal

• Missing value imputation

• Feature transformation (Yeo-Johnson, Log, StandardScaler)

• Class balancing

• Model Building

• XGBoost classifier with hyperparameter tuning

• Evaluation on train/test sets

Data Privacy Note
🚫 The dataset is internal and confidential. Only aggregate profiling results (via pandas_profiling) are included.