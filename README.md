# Customer Churn Prediction

ระบบทำนายการหยุดใช้บริการของลูกค้าธนาคาร (Customer Churn) โดยใช้ Machine Learning

## 📊 ภาพรวมโปรเจกต์

โปรเจกต์นี้ใช้ข้อมูลลูกค้าธนาคารเพื่อทำนายว่าลูกค้าคนไหนมีแนวโน้มจะหยุดใช้บริการ (Churn) โดยใช้:

- **Logistic Regression** (Baseline model)
- **XGBoost** (High-performance model)
- **Hyperparameter Tuning** (Optimized model)
- **Threshold Tuning** (Balanced predictions)
- **SHAP** สำหรับอธิบายผลการทำนาย

## 🚀 Quick Start

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 2. รัน Best Model (Run #2.2 - Hyperparameter Tuned + Threshold 0.54)

```bash
# Train models with optimized hyperparameters
python train_models.py

# สร้าง visualizations
python evaluate_models.py

# วิเคราะห์ด้วย SHAP
python shap_analysis.py
```

### 3. ทดลอง Approaches อื่นๆ (Optional)

```bash
# Hyperparameter Tuning
python hyperparameter_tuning.py

# Threshold Tuning
python threshold_tuning.py

# Cost-Sensitive Learning
python train_models.py --cost-sensitive
```

**หมายเหตุ:** Run #2.2 (Hyperparameter Tuned + Threshold 0.54) ให้ผลดีที่สุด!

## 📁 โครงสร้างโปรเจกต์

```
Customer Churn Prediction/
├── README.md                    # ไฟล์นี้
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration และ hyperparameters
├── logger_config.py             # Logging setup
│
├── data/
│   └── Churn_Modelling.csv     # Dataset (10,000 ลูกค้า)
│
├── feature_binning.py           # Custom transformers สำหรับ binning
├── imbalance_handlers.py        # SMOTE, ADASYN, SMOTETomek handlers
├── cost_sensitive.py            # Cost-sensitive learning utilities
├── data_prep.py                 # Data preparation pipeline
├── train_models.py              # Model training script
├── evaluate_models.py           # Evaluation & visualization
├── shap_analysis.py             # SHAP explainability
├── hyperparameter_tuning.py     # Hyperparameter optimization
├── threshold_tuning.py          # Threshold optimization
│
├── models/                      # Trained models (แยกตาม run)
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
├── plots/                       # Visualizations (แยกตาม run)
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
    ├── runs/                    # รายละเอียดแต่ละ run
    │   ├── README.md
    │   ├── run_01_baseline.md
    │   ├── run_02_class_weights.md
    │   ├── run_02.2_threshold_tuned.md  # ⭐ Best model details
    │   ├── run_03_smote.md
    │   ├── run_04_adasyn.md
    │   ├── run_05_smotetomek.md
    │   └── run_06_cost_sensitive.md
    ├── walkthrough.md           # คู่มือการใช้งานโดยละเอียด
    ├── RESULTS.md               # บันทึกผลการทดลองทั้งหมด
    └── COST_SENSITIVE_GUIDE.md  # คู่มือ Cost-Sensitive Learning
```

## 🎯 Features

### Data Preparation

- ✅ Error handling และ validation
- ✅ DEBUG-level logging
- ✅ Feature binning (Age, CreditScore, Tenure, Balance)
- ✅ **Separate preprocessing pipelines:**
  - **Logistic Regression:** OneHot encoding (25 features)
  - **XGBoost:** Label encoding (10 features) - ดีกว่า OneHot!
- ✅ Train/Val/Test split (70/15/15) แบบ stratified

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

### Imbalance Handling (Tested 5 Approaches)

- ✅ **Class Weights** (Run #2) - Good baseline
- ✅ **Hyperparameter Tuning** (Run #2.1) - High Recall
- ✅ **Threshold Tuning** (Run #2.2) - ⭐ **Most Balanced!**
- ✅ **SMOTE** (Run #3) - Overfitting
- ✅ **ADASYN** (Run #4) - Overfitting
- ✅ **SMOTETomek** (Run #5) - Overfitting
- ✅ **Cost-Sensitive** (Run #6) - Extreme Recall

**สรุป:** Hyperparameter Tuning + Threshold 0.54 ให้ผลดีที่สุด!

### Model Training

- ✅ Logistic Regression with `class_weight='balanced'`
- ✅ XGBoost with optimized hyperparameters
- ✅ 5-Fold Cross-Validation
- ✅ Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- ✅ **Versioned runs** - บันทึกผลทุก experiment

### Evaluation & Explainability

- ✅ Confusion Matrix (แยก LR และ XGB)
- ✅ ROC Curves (เปรียบเทียบ LR vs XGB)
- ✅ Precision-Recall Curves
- ✅ Feature Importance (LR coefficients และ XGB weights)
- ✅ **SHAP Analysis:**
  - Summary Plot - ภาพรวม feature importance
  - Bar Plot - Mean absolute SHAP values
  - Waterfall Plots - อธิบายการทำนายแต่ละลูกค้า
  - Dependence Plot - ความสัมพันธ์ระหว่าง features

## 📊 ผลลัพธ์

ดูผลการทดลองโดยละเอียดได้ที่ [Doc/RESULTS.md](Doc/RESULTS.md)

### 🏆 Best Model: Run #2.2 (Hyperparameter Tuned + Threshold 0.54)

**XGBoost Performance (Test Set):**

| Metric        | Score      | Status                         |
| ------------- | ---------- | ------------------------------ |
| **F1 Score**  | **0.5811** | 🏆 **สูงสุด!**                 |
| **ROC-AUC**   | **0.8461** | ✅ เกินเป้า 0.80               |
| **Recall**    | **0.7026** | ✅ เกินเป้า 0.70               |
| **Precision** | **0.4954** | ✅ สูงสุดในกลุ่ม Recall >= 70% |
| **Accuracy**  | **0.7933** | ✅ ดีมาก                       |

**Optimized Hyperparameters:**

- `n_estimators`: 50
- `max_depth`: 3
- `learning_rate`: 0.1
- `subsample`: 0.6
- `reg_lambda`: 0.1
- `reg_alpha`: 0.5
- `threshold`: 0.54

**Top 3 Features (SHAP Analysis):**

1. **Balance** (0.7238) - ยอดเงินในบัญชี (สำคัญที่สุด!)
2. **NumOfProducts** (0.6868) - จำนวน products (3-4 = Churn สูง, 2 = ดีที่สุด)
3. **IsActiveMember** (0.3250) - ลูกค้า Active หรือไม่ (ไม่ Active = Churn สูงมาก)

### 📊 เปรียบเทียบทั้งหมด (Top 5)

| Run  | Method                         | ROC-AUC    | Recall     | Precision  | F1         | Ranking           |
| ---- | ------------------------------ | ---------- | ---------- | ---------- | ---------- | ----------------- |
| #2.2 | **Hyperparameter + T=0.54** ⭐ | **0.8461** | **0.7026** | **0.4954** | **0.5811** | 🥇 **Best**       |
| #2.1 | Hyperparameter Tuned           | **0.8461** | **0.7451** | 0.4740     | 0.5794     | 🥈 High Recall    |
| #2   | Class Weights                  | 0.8379     | 0.6895     | 0.4862     | 0.5703     | 🥉 Baseline       |
| #6   | Cost-Sensitive                 | 0.8220     | **0.9183** | 0.2838     | 0.4336     | 🎯 Extreme Recall |
| #3   | SMOTE                          | 0.8170     | 0.6144     | 0.5123     | 0.5587     | 4th               |

**💡 Key Findings:**

1. **Hyperparameter Tuning** เพิ่ม Recall จาก 68.95% → 74.51% (+5.56 pp)
2. **Threshold 0.54** ให้ F1 Score สูงสุด (58.11%) และ Balance ดีที่สุด
3. **ROC-AUC = 84.61%** สูงสุด (เกินเป้าหมาย 80%)
4. **Recall = 70.26%** เกินเป้าหมาย 70% พอดี
5. **Synthetic sampling ทุกวิธีสร้าง overfitting** - ไม่แนะนำ!

### 🎯 Business Impact (Run #2.2)

- **ประหยัดได้:** 12.78 ล้านบาท/ปี (จากลูกค้า 2,000 คน)
- **ROI:** 5,789% 🚀 (สูงสุด!)
- **รักษาลูกค้าไว้ได้:** 65 คน (มูลค่า 6.5 ล้านบาท)
- **Churn Rate ลดลง:** จาก 15.3% → 12.1%
- **ต้นทุนต่ำสุด:** 217,000 บาท (ติดต่อ 434 คน)

### 🎯 Model Selection Guide

**สำหรับธนาคาร:**

- **ธนาคารทั่วไป (ต้องการ Balance)** → Run #2.2 ⭐ **แนะนำ!**
  - F1 Score สูงสุด, Balance ดีที่สุด, ROI สูงสุด
- **ธนาคารที่ต้องการ Recall สูง** → Run #2.1 🚀
  - Recall = 74.51%, ROC-AUC = 84.61%
- **ธนาคารที่ต้องการ Simplicity** → Run #2
  - ใช้ default hyperparameters, ผลลัพธ์ดี
- **Campaign พิเศษ (ยอมรับ False Positive สูง)** → Run #6 🎯
  - Recall = 91.83% (สูงสุด!)

## 📖 เอกสาร

- **[Walkthrough](Doc/walkthrough.md)** - คู่มือการใช้งานโดยละเอียด
- **[Results](Doc/RESULTS.md)** - ผลการทดลองและ metrics แต่ละรอบ
- **[Run #2.2 Details](Doc/runs/run_02.2_threshold_tuned.md)** - รายละเอียด Best Model
- **[Cost-Sensitive Guide](Doc/COST_SENSITIVE_GUIDE.md)** - คู่มือ Cost-Sensitive Learning

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
