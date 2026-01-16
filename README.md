# Customer Churn Prediction

A Machine Learning system to predict customer churn for banking services

## 📊 Project Overview

This project uses bank customer data to predict which customers are likely to churn (stop using services) using:

- **Logistic Regression** (Baseline model)
- **XGBoost** (High-performance model)
- **Hyperparameter Tuning** (Optimized model)
- **Threshold Tuning** (Balanced predictions)
- **SHAP** for model interpretability

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Best Model (Run #2.2 - Hyperparameter Tuned + Threshold 0.54)

```bash
# Train models with optimized hyperparameters
python train_models.py

# Generate visualizations
python evaluate_models.py

# Analyze with SHAP
python shap_analysis.py
```

### 3. Try Other Approaches (Optional)

```bash
# Hyperparameter Tuning
python hyperparameter_tuning.py

# Threshold Tuning
python threshold_tuning.py

# Cost-Sensitive Learning
python train_models.py --cost-sensitive
```

**Note:** Run #2.2 (Hyperparameter Tuned + Threshold 0.54) gives the best results!

## 📁 Project Structure

```
Customer Churn Prediction/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration and hyperparameters
├── logger_config.py             # Logging setup
│
├── data/
│   └── Churn_Modelling.csv     # Dataset (10,000 customers)
│
├── feature_binning.py           # Custom transformers for binning
├── imbalance_handlers.py        # SMOTE, ADASYN, SMOTETomek handlers
├── cost_sensitive.py            # Cost-sensitive learning utilities
├── data_prep.py                 # Data preparation pipeline
├── train_models.py              # Model training script
├── evaluate_models.py           # Evaluation & visualization
├── shap_analysis.py             # SHAP explainability
├── hyperparameter_tuning.py     # Hyperparameter optimization
├── threshold_tuning.py          # Threshold optimization
│
├── models/                      # Trained models (separated by run)
│   ├── run_1/                   # Baseline (OneHot for both)
│   ├── run_2/                   # Separate Preprocessing
│   ├── run_2.2/                 # ⭐ Best! (Hyperparameter + Threshold)
│   ├── run_2_tuned/             # Hyperparameter tuning results
│   ├── run_3/                   # SMOTE Resampling
│   ├── run_4/                   # ADASYN Resampling
│   ├── run_5/                   # SMOTETomek Resampling
│   └── run_6/                   # Cost-Sensitive Learning
│       ├── logistic_regression.pkl
│       ├── xgboost.pkl
│       ├── preprocessor_lr.pkl
│       └── preprocessor_xgb.pkl
│
├── plots/                       # Visualizations (separated by run)
│   ├── run_1/
│   ├── run_2/
│   ├── run_2.2/                 # ⭐ Best model visualizations
│   ├── run_3/
│   ├── run_4/
│   ├── run_5/
│   └── run_6/
│       ├── confusion_matrix_lr.png
│       ├── confusion_matrix_xgb.png
│       ├── roc_curves.png
│       ├── precision_recall_curves.png
│       ├── feature_importance_lr.png
│       ├── feature_importance_xgb.png
│       ├── shap_summary.png
│       ├── shap_bar.png
│       ├── shap_waterfall_sample0.png
│       ├── shap_waterfall_churn.png
│       └── shap_dependence_top.png
│
├── experiments/                 # Experiment results
│   └── run_2.1_threshold_tuning/
│       ├── threshold_results.csv
│       └── threshold_tuning_analysis.png
│
└── Doc/
    ├── runs/                    # Detailed run documentation
    │   ├── README.md
    │   ├── run_01_baseline.md
    │   ├── run_02_class_weights.md
    │   ├── run_02.2_threshold_tuned.md  # ⭐ Best model details
    │   ├── run_03_smote.md
    │   ├── run_04_adasyn.md
    │   ├── run_05_smotetomek.md
    │   └── run_06_cost_sensitive.md
    ├── walkthrough.md           # Detailed user guide
    ├── RESULTS.md               # All experiment results
    └── COST_SENSITIVE_GUIDE.md  # Cost-Sensitive Learning guide
```

## 🎯 Features

### Data Preparation

- ✅ Error handling and validation
- ✅ DEBUG-level logging
- ✅ Feature binning (Age, CreditScore, Tenure, Balance)
- ✅ **Separate preprocessing pipelines:**
  - **Logistic Regression:** OneHot encoding (25 features)
  - **XGBoost:** Label encoding (10 features) - Better than OneHot!
- ✅ Train/Val/Test split (70/15/15) with stratification

### Model Optimization

- ✅ **Hyperparameter Tuning** (Run #2.1)
  - RandomizedSearchCV with 50 iterations
  - Custom scorer (Recall 60% + F1 40%)
  - Best params: n_estimators=50, max_depth=3, learning_rate=0.1
- ✅ **Threshold Tuning** (Run #2.2)

  - Tested thresholds from 0.1 to 0.99
  - Optimal threshold: 0.54 for best balance
  - Maximizes F1 Score while maintaining Recall >= 70%

- ✅ **Cost-Sensitive Learning** (Run #6)
  - Sample weighting for imbalanced data
  - Extreme Recall (91.83%) for special campaigns

### Imbalance Handling (Tested 7 Approaches)

- ✅ **Class Weights** (Run #2) - Good baseline
- ✅ **Hyperparameter Tuning** (Run #2.1) - High Recall
- ✅ **Threshold Tuning** (Run #2.2) - ⭐ **Most Balanced!**
- ✅ **SMOTE** (Run #3) - Overfitting
- ✅ **ADASYN** (Run #4) - Overfitting
- ✅ **SMOTETomek** (Run #5) - Overfitting
- ✅ **Cost-Sensitive** (Run #6) - Extreme Recall

**Conclusion:** Hyperparameter Tuning + Threshold 0.54 gives the best results!

### Model Training

- ✅ Logistic Regression with `class_weight='balanced'`
- ✅ XGBoost with optimized hyperparameters
- ✅ 5-Fold Cross-Validation
- ✅ Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- ✅ **Versioned runs** - Track all experiments

### Evaluation & Explainability

- ✅ Confusion Matrix (separate for LR and XGB)
- ✅ ROC Curves (compare LR vs XGB)
- ✅ Precision-Recall Curves
- ✅ Feature Importance (LR coefficients and XGB weights)
- ✅ **SHAP Analysis:**
  - Summary Plot - Overall feature importance
  - Bar Plot - Mean absolute SHAP values
  - Waterfall Plots - Explain individual predictions
  - Dependence Plot - Feature relationships

## 📊 Results

See detailed experiment results at [Doc/RESULTS.md](Doc/RESULTS.md)

### 🏆 Best Model: Run #2.2 (Hyperparameter Tuned + Threshold 0.54)

**XGBoost Performance (Test Set):**

| Metric        | Score      | Status                            |
| ------------- | ---------- | --------------------------------- |
| **F1 Score**  | **0.5811** | 🏆 **Highest!**                   |
| **ROC-AUC**   | **0.8461** | ✅ Exceeds target 0.80            |
| **Recall**    | **0.7026** | ✅ Exceeds target 0.70            |
| **Precision** | **0.4954** | ✅ Highest in Recall >= 70% group |
| **Accuracy**  | **0.7933** | ✅ Excellent                      |

**Optimized Hyperparameters:**

- `n_estimators`: 50
- `max_depth`: 3
- `learning_rate`: 0.1
- `subsample`: 0.6
- `reg_lambda`: 0.1
- `reg_alpha`: 0.5
- `threshold`: 0.54

**Top 3 Features (SHAP Analysis):**

1. **Balance** (0.7238) - Account balance (Most important!)
2. **NumOfProducts** (0.6868) - Number of products (3-4 = High churn, 2 = Best)
3. **IsActiveMember** (0.3250) - Active customer status (Inactive = High churn)

### 📊 Comparison (Top 5 Runs)

| Run  | Method                         | ROC-AUC    | Recall     | Precision  | F1         | Ranking           |
| ---- | ------------------------------ | ---------- | ---------- | ---------- | ---------- | ----------------- |
| #2.2 | **Hyperparameter + T=0.54** ⭐ | **0.8461** | **0.7026** | **0.4954** | **0.5811** | 🥇 **Best**       |
| #2.1 | Hyperparameter Tuned           | **0.8461** | **0.7451** | 0.4740     | 0.5794     | 🥈 High Recall    |
| #2   | Class Weights                  | 0.8379     | 0.6895     | 0.4862     | 0.5703     | 🥉 Baseline       |
| #6   | Cost-Sensitive                 | 0.8220     | **0.9183** | 0.2838     | 0.4336     | 🎯 Extreme Recall |
| #3   | SMOTE                          | 0.8170     | 0.6144     | 0.5123     | 0.5587     | 4th               |

**💡 Key Findings:**

1. **Hyperparameter Tuning** increased Recall from 68.95% → 74.51% (+5.56 pp)
2. **Threshold 0.54** achieves highest F1 Score (58.11%) with best balance
3. **ROC-AUC = 84.61%** - Highest (exceeds 80% target)
4. **Recall = 70.26%** - Exceeds 70% target perfectly
5. **Synthetic sampling causes overfitting** - Not recommended!

### 🎯 Business Impact (Run #2.2)

- **Cost Savings:** 12.78M THB/year (from 2,000 customers)
- **ROI:** 5,789% 🚀 (Highest!)
- **Customers Retained:** 65 customers (worth 6.5M THB)
- **Churn Rate Reduction:** From 15.3% → 12.1%
- **Lowest Cost:** 217,000 THB (contact 434 customers)

### 🎯 Model Selection Guide

**For Banks:**

- **General Banks (Need Balance)** → Run #2.2 ⭐ **Recommended!**
  - Highest F1 Score, Best balance, Highest ROI
- **Banks Needing High Recall** → Run #2.1 🚀
  - Recall = 74.51%, ROC-AUC = 84.61%
- **Banks Needing Simplicity** → Run #2
  - Uses default hyperparameters, good results
- **Special Campaigns (Accept High False Positives)** → Run #6 🎯
  - Recall = 91.83% (Highest!)

## 📖 Documentation

- **[Walkthrough](Doc/walkthrough.md)** - Detailed user guide
- **[Results](Doc/RESULTS.md)** - All experiment results and metrics
- **[Run #2.2 Details](Doc/runs/run_02.2_threshold_tuned.md)** - Best model details
- **[Cost-Sensitive Guide](Doc/COST_SENSITIVE_GUIDE.md)** - Cost-Sensitive Learning guide

## 🛠️ Technologies

- Python 3.12
- **Machine Learning:**
  - scikit-learn - Logistic Regression, preprocessing, GridSearchCV
  - XGBoost - Gradient boosting with hyperparameter tuning
  - imbalanced-learn - SMOTE, ADASYN, SMOTETomek
- **Explainability:**
  - SHAP - Model interpretation
- **Data Processing:**
  - pandas, numpy
- **Visualization:**
  - matplotlib, seaborn

## 📝 License

This project is for educational purposes.

## 👤 Author

Created as part of a Customer Churn Prediction project.

**Last Updated:** 2026-01-16

**Total Experiments:** 8 Runs (6 main + Hyperparameter Tuning + Threshold Tuning)

**Best Model:** Run #2.2 - Hyperparameter Tuned + Threshold 0.54 ⭐
