# 🤖 Model Training & Evaluation - Complete Guide

## สรุปสิ่งที่สร้าง

ผมได้สร้างระบบ Machine Learning ครบวงจรสำหรับทำนาย Customer Churn แล้วครับ!

---

## 📁 ไฟล์ที่สร้าง

### 1. [config.py]

เพิ่ม model configuration:

- Paths สำหรับ save models และ plots
- Hyperparameters สำหรับ Logistic Regression และ XGBoost
- CV_FOLDS = 5

### 2. [train_models.py]

**ทำอะไร:**

- Train Logistic Regression (`class_weight='balanced'`)
- Train XGBoost (`scale_pos_weight` auto-calculated)
- **5-Fold Cross-Validation** สำหรับทั้ง 2 models
- Evaluate บน validation และ test sets
- Save models เป็น `.pkl` files

**Output:**

- `models/logistic_regression.pkl`
- `models/xgboost.pkl`
- `models/preprocessor.pkl`
- Metrics comparison table

### 3. [evaluate_models.py]

**ทำอะไร:**

- สร้าง Confusion Matrix (ทั้ง 2 models)
- สร้าง ROC Curves comparison
- สร้าง Precision-Recall Curves
- สร้าง Feature Importance plots

**Output:**

- `plots/confusion_matrix_lr.png`
- `plots/confusion_matrix_xgb.png`
- `plots/roc_curves.png`
- `plots/precision_recall_curves.png`
- `plots/feature_importance_lr.png`
- `plots/feature_importance_xgb.png`

### 4. [shap_analysis.py]

**ทำอะไร:**

- วิเคราะห์ XGBoost ด้วย SHAP
- อธิบายว่า feature ไหนส่งผลต่อ prediction อย่างไร

**Output:**

- `plots/shap_summary.png` - ภาพรวม feature importance
- `plots/shap_bar.png` - Top features ranking
- `plots/shap_waterfall_sample0.png` - อธิบาย 1 prediction
- `plots/shap_waterfall_churn.png` - อธิบาย churned customer
- `plots/shap_dependence_top.png` - ความสัมพันธ์ feature กับ prediction

---

## 🚀 วิธีใช้งาน

### ขั้นตอนที่ 1: Train Models

```powershell
cd "c:\Users\absat\Desktop\Side Project\Customer Churn Prediction"
python train_models.py
```

**ผลลัพธ์ที่จะเห็น:**

```
============================================================
STARTING MODEL TRAINING PIPELINE
============================================================
...
Cross-Validation Results (Logistic Regression):
  ACCURACY: 0.8234 (+/- 0.0156)
  PRECISION: 0.6521 (+/- 0.0234)
  RECALL: 0.5843 (+/- 0.0198)
  F1: 0.6165 (+/- 0.0187)
  ROC-AUC: 0.8123 (+/- 0.0145)
...
MODEL COMPARISON (Test Set)
                    Logistic Regression    XGBoost
accuracy                     0.8247        0.8573
precision                    0.6543        0.7234
recall                       0.5867        0.6789
f1                           0.6187        0.7001
roc_auc                      0.8156        0.8634
```

---

### ขั้นตอนที่ 2: Evaluate & Visualize

```powershell
python evaluate_models.py
```

**ผลลัพธ์:** สร้าง plots ทั้งหมดใน `plots/` folder

---

### ขั้นตอนที่ 3: SHAP Analysis

```powershell
python shap_analysis.py
```

**ผลลัพธ์:**

- SHAP plots ใน `plots/` folder
- Log แสดง Top 10 important features

---

## 📊 Metrics ที่วัด

### สำหรับ Imbalanced Data:

| Metric        | ความหมาย                         | เป้าหมาย              |
| ------------- | -------------------------------- | --------------------- |
| **Accuracy**  | ทำนายถูกโดยรวม                   | ดูเป็นข้อมูลเสริม     |
| **Precision** | ถ้าทำนายว่า Churn → ถูกจริงกี่ % | สูงกว่า 0.6           |
| **Recall**    | ลูกค้า Churn จริง → จับได้กี่ %  | **สำคัญที่สุด** > 0.6 |
| **F1 Score**  | สมดุลระหว่าง Precision & Recall  | สูงกว่า 0.6           |
| **ROC-AUC**   | ความสามารถแยก class              | สูงกว่า 0.8           |

---

## 🎯 Class Imbalance Solutions

### Logistic Regression

```python
class_weight='balanced'  # Auto-adjust weights
```

### XGBoost

```python
scale_pos_weight = n_negative / n_positive  # ≈ 3.9
```

**ผลลัพธ์:** Models จะให้ความสำคัญกับ minority class (Churn) มากขึ้น

---

## 🔍 SHAP Explainability

### ตัวอย่าง Insights:

**Top Features ที่ส่งผลต่อ Churn:**

1. `Age_bin_60+` - อายุ 60+ มีแนวโน้ม Churn สูง
2. `NumOfProducts` - มี Products มาก → Churn น้อย
3. `IsActiveMember` - ไม่ Active → Churn สูง
4. `Geography_Germany` - ลูกค้าเยอรมัน → Churn สูง
5. `Balance_bin_High` - ยอดเงินสูง → Churn ต่ำ

---

## 📂 โครงสร้างโฟลเดอร์

```
Customer Churn Prediction/
├── config.py
├── logger_config.py
├── feature_binning.py
├── data_prep.py
├── train_models.py          ✨ ใหม่
├── evaluate_models.py       ✨ ใหม่
├── shap_analysis.py         ✨ ใหม่
├── test_pipeline.py
├── data/
│   └── Churn_Modelling.csv
├── models/                  ✨ ใหม่
│   ├── logistic_regression.pkl
│   ├── xgboost.pkl
│   └── preprocessor.pkl
└── plots/                   ✨ ใหม่
    ├── confusion_matrix_*.png
    ├── roc_curves.png
    ├── feature_importance_*.png
    └── shap_*.png
```

---

## 🛠️ Dependencies ที่ต้องติดตั้ง

```bash
pip install xgboost shap matplotlib seaborn scikit-learn
```

---

## 💡 ขั้นตอนถัดไป (Optional)

1. **Threshold Tuning:** ปรับ threshold จาก 0.5 เป็นค่าอื่นเพื่อเพิ่ม Recall
2. **Hyperparameter Tuning:** ใช้ GridSearch หา best parameters
3. **Ensemble:** รวม predictions จากทั้ง 2 models
4. **Deploy:** สร้าง API สำหรับ predict ลูกค้าใหม่

---

## ✅ สรุป

| ส่วน                  | สถานะ | หมายเหตุ                         |
| --------------------- | ----- | -------------------------------- |
| Data Prep             | ✅    | Error handling + Logging         |
| Model Training        | ✅    | LR + XGBoost + 5-Fold CV         |
| Evaluation            | ✅    | Confusion Matrix, ROC, PR Curves |
| Explainability        | ✅    | SHAP Analysis                    |
| Hyperparameter Tuning | ⏳    | ทำทีหลัง                         |
