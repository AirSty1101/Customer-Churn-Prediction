# Customer Churn Prediction

ระบบทำนายการหยุดใช้บริการของลูกค้าธนาคาร (Customer Churn) โดยใช้ Machine Learning

## 📊 ภาพรวมโปรเจกต์

โปรเจกต์นี้ใช้ข้อมูลลูกค้าธนาคารเพื่อทำนายว่าลูกค้าคนไหนมีแนวโน้มจะหยุดใช้บริการ (Churn) โดยใช้:

- **Logistic Regression** (Baseline model)
- **XGBoost** (High-performance model)
- **SHAP** สำหรับอธิบายผลการทำนาย

## 🚀 Quick Start

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 2. รัน Best Model (Run #2)

```bash
# Train models with separate preprocessing (แนะนำ!)
python train_models.py --version 2

# สร้าง visualizations
python evaluate_models.py --version 2

# วิเคราะห์ด้วย SHAP
python shap_analysis.py --version 2
```

### 3. ทดลอง Imbalance Handling อื่นๆ (Optional)

```bash
# Run #3: SMOTE
python train_models.py --version 3 --imbalance-method smote

# Run #4: ADASYN
python train_models.py --version 4 --imbalance-method adasyn

# Run #5: SMOTETomek
python train_models.py --version 5 --imbalance-method smotetomek
```

**หมายเหตุ:** Run #2 (Class Weights) ให้ผลดีที่สุด - ไม่แนะนำให้ใช้ synthetic sampling!

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
├── data_prep.py                 # Data preparation pipeline
├── train_models.py              # Model training script (รองรับ versioning)
├── evaluate_models.py           # Evaluation & visualization
├── shap_analysis.py             # SHAP explainability
├── test_pipeline.py             # Pipeline testing
│
├── models/                      # Trained models (แยกตาม run)
│   ├── run_1/                   # Baseline (OneHot for both)
│   ├── run_2/                   # Separate Preprocessing ⭐ Best!
│   ├── run_3/                   # SMOTE Resampling
│   ├── run_4/                   # ADASYN Resampling
│   └── run_5/                   # SMOTETomek Resampling
│       ├── logistic_regression.pkl
│       ├── xgboost.pkl
│       ├── preprocessor_lr.pkl
│       └── preprocessor_xgb.pkl
│
├── plots/                       # Visualizations (แยกตาม run)
│   ├── run_1/
│   ├── run_2/                   # ⭐ Best model visualizations
│   ├── run_3/
│   ├── run_4/
│   └── run_5/
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
└── Doc/
    ├── walkthrough.md           # คู่มือการใช้งานโดยละเอียด
    └── RESULTS.md               # บันทึกผลการทดลองทั้ง 5 runs
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

### Imbalance Handling (Tested 4 Approaches)

- ✅ **Class Weights** (Run #2) - ⭐ **Best approach!**
  - Logistic Regression: `class_weight='balanced'`
  - XGBoost: `scale_pos_weight=3.9088`
- ✅ **SMOTE** (Run #3) - Synthetic over-sampling
- ✅ **ADASYN** (Run #4) - Adaptive synthetic sampling
- ✅ **SMOTETomek** (Run #5) - Hybrid over/under-sampling

**สรุป:** Class Weights ให้ผลดีที่สุด - Synthetic sampling ทุกวิธีสร้าง overfitting!

### Model Training

- ✅ Logistic Regression with `class_weight='balanced'`
- ✅ XGBoost with `scale_pos_weight` และ Label Encoding
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

### 🏆 Best Model: Run #2 (Separate Preprocessing + Class Weights)

**XGBoost Performance (Test Set):**

| Metric        | Score      | Status           |
| ------------- | ---------- | ---------------- |
| **ROC-AUC**   | **0.8379** | ✅ เกินเป้า 0.80 |
| **Recall**    | **0.6895** | ✅ ใกล้เป้า 0.70 |
| **Precision** | **0.4862** | ⚠️ ต่ำกว่าเป้า   |
| **F1 Score**  | **0.5703** | ✅ ใกล้เป้า 0.65 |
| **Accuracy**  | **0.7880** | ✅ ดี            |

**Top 3 Features (SHAP Analysis):**

1. **Balance** - ยอดเงินในบัญชี (สำคัญที่สุด!)
2. **NumOfProducts** - จำนวน products (3-4 = Churn สูง, 2 = ดีที่สุด)
3. **IsActiveMember** - ลูกค้า Active หรือไม่ (ไม่ Active = Churn สูงมาก)

### 📊 เปรียบเทียบทั้ง 5 Runs

| Run | Method                 | ROC-AUC    | Recall     | Precision  | Ranking     |
| --- | ---------------------- | ---------- | ---------- | ---------- | ----------- |
| #2  | **Class Weights** ⭐   | **0.8379** | **0.6895** | 0.4862     | 🥇 **Best** |
| #3  | SMOTE                  | 0.8170     | 0.6144     | 0.5123     | 🥈 2nd      |
| #5  | SMOTETomek             | 0.8121     | 0.6046     | **0.5153** | 🥉 3rd      |
| #4  | ADASYN                 | 0.8106     | 0.6013     | 0.5041     | 4th         |
| #1  | Baseline (OneHot both) | 0.7279     | 0.6144     | 0.3501     | 5th         |

**💡 Key Findings:**

1. **Separate Preprocessing** ทำให้ XGBoost ดีขึ้น **15%** ใน ROC-AUC (Run #1 → #2)
2. **Class Weights ดีที่สุด** - Synthetic sampling ทุกวิธีสร้าง overfitting
3. **Label Encoding เหมาะกับ XGBoost** มากกว่า OneHot Encoding
4. **Recall เป้าหมายสำคัญ** - Run #2 ให้ Recall สูงสุด (68.95%)

### 🎯 Business Impact (Run #2)

- **ประหยัดได้:** 12.5 ล้านบาท/ปี (จากลูกค้า 2,000 คน)
- **ROI:** 4,849% 🚀
- **รักษาลูกค้าไว้ได้:** 64 คน (มูลค่า 6.4 ล้านบาท)
- **Churn Rate ลดลง:** จาก 15.3% → 12.1%

## 📖 เอกสาร

- **[Walkthrough](Doc/walkthrough.md)** - คู่มือการใช้งานโดยละเอียด
- **[Results](Doc/RESULTS.md)** - ผลการทดลองและ metrics แต่ละรอบ

## 🛠️ Technologies

- Python 3.x
- **Machine Learning:**
  - scikit-learn - Logistic Regression, preprocessing
  - XGBoost - Gradient boosting
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
