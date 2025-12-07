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

### 2. รัน Pipeline

```bash
# Train models (รวม 5-Fold Cross-Validation)
python train_models.py

# สร้าง visualizations (Confusion Matrix, ROC Curves, Feature Importance)
python evaluate_models.py

# วิเคราะห์ด้วย SHAP
python shap_analysis.py
```

## 📁 โครงสร้างโปรเจกต์

```
Customer Churn Prediction/
├── README.md                    # ไฟล์นี้
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration ทั้งหมด
├── logger_config.py             # Logging setup
│
├── data/
│   └── Churn_Modelling.csv     # Dataset
│
├── feature_binning.py           # Custom transformer สำหรับ binning
├── data_prep.py                 # Data preparation pipeline
├── train_models.py              # Model training script
├── evaluate_models.py           # Evaluation & visualization
├── shap_analysis.py             # SHAP explainability
│
├── models/                      # Trained models (generated)
│   ├── logistic_regression.pkl
│   ├── xgboost.pkl
│   └── preprocessor.pkl
│
├── plots/                       # Visualizations (generated)
│   ├── confusion_matrix_*.png
│   ├── roc_curves.png
│   ├── feature_importance_*.png
│   └── shap_*.png
│
└── Doc/
    ├── walkthrough.md           # คู่มือการใช้งานโดยละเอียด
    └── RESULTS.md               # บันทึกผลการทดลอง
```

## 🎯 Features

### Data Preparation

- ✅ Error handling และ validation
- ✅ DEBUG-level logging
- ✅ Feature binning (Age, CreditScore, Tenure, Balance)
- ✅ OneHot encoding สำหรับ categorical features
- ✅ Train/Val/Test split (70/15/15) แบบ stratified

### Model Training

- ✅ Logistic Regression with `class_weight='balanced'`
- ✅ XGBoost with `scale_pos_weight`
- ✅ 5-Fold Cross-Validation
- ✅ Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)

### Evaluation & Explainability

- ✅ Confusion Matrix
- ✅ ROC Curves
- ✅ Precision-Recall Curves
- ✅ Feature Importance
- ✅ SHAP Analysis (Summary, Waterfall, Dependence plots)

## 📊 ผลลัพธ์

ดูผลการทดลองโดยละเอียดได้ที่ [Doc/RESULTS.md](Doc/RESULTS.md)

**สรุปสั้นๆ:**

- XGBoost ให้ผลลัพธ์ดีกว่า Logistic Regression
- ROC-AUC > 0.86 (XGBoost)
- Top features: Age, NumOfProducts, IsActiveMember

## 📖 เอกสาร

- **[Walkthrough](Doc/walkthrough.md)** - คู่มือการใช้งานโดยละเอียด
- **[Results](Doc/RESULTS.md)** - ผลการทดลองและ metrics แต่ละรอบ

## 🛠️ Technologies

- Python 3.x
- scikit-learn
- XGBoost
- SHAP
- pandas, numpy
- matplotlib, seaborn

## 📝 License

This project is for educational purposes.

## 👤 Author

Created as part of a Customer Churn Prediction project.
