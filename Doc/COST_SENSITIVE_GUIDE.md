# Cost-Sensitive Learning Implementation Guide

## 📋 Overview

Cost-Sensitive Learning เป็นเทคนิคที่กำหนด **cost (ต้นทุน)** ที่แตกต่างกันสำหรับ errors แต่ละประเภท เพื่อให้ model focus ที่การลด error ที่มี cost สูง

### ทำไมต้องใช้ Cost-Sensitive Learning?

ในปัญหา Customer Churn Prediction:

- **False Negative (พลาด Churn)** = cost สูงมาก (สูญเสียลูกค้า = สูญเสียรายได้)
- **False Positive (ทำนาย Churn ผิด)** = cost ต่ำ (แค่เสียต้นทุนติดต่อลูกค้า)

**ตัวอย่าง:**

- พลาดลูกค้า 1 คนที่จะ Churn = สูญเสีย 100,000 บาท
- ติดต่อลูกค้าผิด 1 คน = เสียต้นทุน 500 บาท
- **Cost Ratio = 100,000 / 500 = 200** (แต่ในทางปฏิบัติใช้ 5-20 ก็เพียงพอ)

---

## 🎯 Cost-Sensitive Learning vs Other Methods

| Method             | Approach                              | Pros                                                                | Cons                                 |
| ------------------ | ------------------------------------- | ------------------------------------------------------------------- | ------------------------------------ |
| **Class Weights**  | ปรับน้ำหนักของ class ใน loss function | ✅ ง่าย<br>✅ ไม่เปลี่ยนข้อมูล                                      | ⚠️ น้ำหนักคงที่สำหรับทุก sample      |
| **SMOTE/ADASYN**   | สร้าง synthetic samples               | ✅ Balance dataset                                                  | ❌ Overfitting<br>❌ สร้างข้อมูลปลอม |
| **Cost-Sensitive** | กำหนด cost ต่าง sample                | ✅ Flexible<br>✅ สะท้อนความสำคัญทางธุรกิจ<br>✅ ไม่สร้างข้อมูลปลอม | ⚠️ ต้องกำหนด cost ที่เหมาะสม         |

---

## 🔧 Implementation

### 1. Configuration (`config.py`)

```python
# === Cost-Sensitive Learning ===
USE_COST_SENSITIVE = True   # เปิดใช้งาน cost-sensitive learning
COST_RATIO = 10.0           # น้ำหนักของ minority class (Churn)
                            # ค่าที่แนะนำ: 5.0, 10.0, 15.0, 20.0
```

### 2. Create Sample Weights (`cost_sensitive.py`)

```python
from cost_sensitive import get_sample_weights

# สร้าง sample weights
sample_weights = get_sample_weights(
    y_train,
    method='cost_ratio',
    cost_ratio=10.0
)

# ผลลัพธ์:
# - Not Churn (0): weight = 1.0
# - Churn (1): weight = 10.0
```

### 3. Train with Sample Weights

```python
# XGBoost รองรับ sample_weight
xgb_model.fit(
    X_train, y_train,
    sample_weight=sample_weights
)
```

---

## 📊 How It Works

### ตัวอย่าง: Cost Ratio = 10.0

**ข้อมูล:**

- Training samples: 1,000 คน
  - Not Churn (0): 800 คน → weight = 1.0 each
  - Churn (1): 200 คน → weight = 10.0 each

**Total Weight:**

- Not Churn: 800 × 1.0 = 800
- Churn: 200 × 10.0 = 2,000
- **Total: 2,800**

**ผลกระทบ:**

- Model จะให้ความสำคัญกับ Churn samples มากกว่า Not Churn ถึง **10 เท่า**
- ถ้า model ทำนาย Churn ผิด (False Negative) → loss สูงมาก
- Model จะพยายาม**ลด False Negative** มากขึ้น → **Recall เพิ่มขึ้น**

---

## 🧪 Experiment: Finding Optimal Cost Ratio

### วิธีใช้งาน

```bash
# ทดสอบ cost ratios ต่างๆ
python experiment_cost_sensitive.py
```

สคริปต์นี้จะทดสอบ cost ratios: **5.0, 10.0, 15.0, 20.0, 25.0**

### ผลลัพธ์ที่คาดหวัง

| Cost Ratio     | Accuracy | Precision | Recall | F1     | ROC-AUC |
| -------------- | -------- | --------- | ------ | ------ | ------- |
| 0.0 (Baseline) | 0.7880   | 0.4862    | 0.6895 | 0.5703 | 0.8379  |
| 5.0            | 0.7800   | 0.4700    | 0.7200 | 0.5700 | 0.8350  |
| 10.0           | 0.7700   | 0.4500    | 0.7500 | 0.5650 | 0.8300  |
| 15.0           | 0.7600   | 0.4300    | 0.7800 | 0.5600 | 0.8250  |
| 20.0           | 0.7500   | 0.4100    | 0.8000 | 0.5500 | 0.8200  |

**Pattern:**

- ยิ่ง Cost Ratio สูง → **Recall เพิ่ม**, **Precision ลด**
- Trade-off ระหว่าง Precision และ Recall

---

## 🎯 Choosing Optimal Cost Ratio

### วิธีเลือก Cost Ratio

**1. Business-Driven Approach (แนะนำ)**

```python
# คำนวณจาก cost จริง
cost_false_negative = 100_000  # สูญเสียลูกค้า
cost_false_positive = 500      # ต้นทุนติดต่อลูกค้า

cost_ratio = cost_false_negative / cost_false_positive
# = 100,000 / 500 = 200

# แต่ในทางปฏิบัติ ใช้ 5-20 ก็เพียงพอ
# เพราะ cost_ratio สูงเกินไปจะทำให้ Precision ต่ำมาก
```

**2. Metric-Driven Approach**

- ถ้าต้องการ **Recall ≥ 70%** → ลอง cost_ratio = 10-15
- ถ้าต้องการ **F1 Score สูงสุด** → ลอง cost_ratio = 5-10
- ถ้าต้องการ **Balance Precision & Recall** → ลอง cost_ratio = 5-8

**3. Validation-Based Approach**

- ทดสอบหลาย cost ratios (5, 10, 15, 20)
- เลือก cost ratio ที่ให้ **Validation Metrics** ดีที่สุด

---

## 📈 Expected Results

### Comparison: Class Weights vs Cost-Sensitive

**Baseline (Class Weights - Run #2):**

- Accuracy: 0.7880
- Precision: 0.4862
- Recall: 0.6895
- F1: 0.5703
- ROC-AUC: 0.8379

**Cost-Sensitive (Cost Ratio = 10.0 - Expected):**

- Accuracy: 0.7700 (-1.8%)
- Precision: 0.4500 (-3.6%)
- Recall: **0.7500 (+6.0%)** ✅
- F1: 0.5650 (-0.5%)
- ROC-AUC: 0.8300 (-0.8%)

**Key Takeaway:**

- ✅ **Recall เพิ่มขึ้น 6%** (จาก 68.95% → 75%)
- ⚠️ Precision ลดลงเล็กน้อย (จาก 48.62% → 45%)
- ⚠️ ROC-AUC ลดลงเล็กน้อย (จาก 0.8379 → 0.83)

---

## 💡 Recommendations

### เมื่อไหร่ควรใช้ Cost-Sensitive Learning?

✅ **ควรใช้เมื่อ:**

1. **False Negative มี cost สูงกว่า False Positive มาก**
   - เช่น: Customer Churn, Fraud Detection, Medical Diagnosis
2. **ต้องการเพิ่ม Recall** โดยเฉพาะ
3. **ยอมรับ Precision ลดลงได้**
4. **มีข้อมูล cost จริงจากธุรกิจ**

❌ **ไม่ควรใช้เมื่อ:**

1. **Precision สำคัญมากกว่า Recall**
2. **ต้องการ Balance ระหว่าง Precision & Recall**
3. **ไม่มีข้อมูล cost ที่ชัดเจน**

### Alternative: Threshold Tuning

ถ้าไม่แน่ใจว่า cost ratio ควรเป็นเท่าไร → ลอง **Threshold Tuning** ก่อน

```python
# แทนที่จะใช้ cost-sensitive
# ลองปรับ threshold แทน
y_pred = (y_pred_proba >= 0.4).astype(int)  # ลด threshold จาก 0.5 → 0.4
# → Recall เพิ่มขึ้น, Precision ลดลง
```

**ข้อดี:**

- ✅ ง่ายกว่า (ไม่ต้อง retrain model)
- ✅ ทดสอบได้เร็ว
- ✅ ปรับได้ตลอดเวลา

---

## 🚀 Quick Start

### Run #6: Cost-Sensitive Learning

**1. Update config.py:**

```python
RUN_NUMBER = 6
RESAMPLING_METHOD = 'none'
USE_COST_SENSITIVE = True
COST_RATIO = 10.0
```

**2. Train model:**

```bash
python train_models.py
```

**3. (Optional) Experiment with different cost ratios:**

```bash
python experiment_cost_sensitive.py
```

---

## 📝 Notes

### Limitations

1. **Cross-Validation ไม่รองรับ sample_weight**

   - `sklearn.cross_validate` ไม่รองรับ `sample_weight` สำหรับ XGBoost
   - ต้อง skip CV หรือใช้ manual CV

2. **Hyperparameter Tuning ยากขึ้น**

   - ต้อง tune ทั้ง XGBoost parameters และ cost_ratio
   - เพิ่มความซับซ้อน

3. **อาจ Overfit ได้**
   - ถ้า cost_ratio สูงเกินไป model อาจ overfit ไปที่ minority class

### Best Practices

1. **เริ่มจาก cost_ratio ต่ำๆ** (5-10) แล้วค่อยเพิ่ม
2. **Monitor Validation Metrics** เพื่อดู overfitting
3. **เปรียบเทียบกับ Baseline** (Class Weights)
4. **ใช้ Business Metrics** (ROI, Cost Savings) ในการตัดสินใจ

---

## 📚 References

- [XGBoost Documentation - Sample Weight](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier.fit)
- [Cost-Sensitive Learning for Imbalanced Classification](https://machinelearningmastery.com/cost-sensitive-learning-for-imbalanced-classification/)
- [Scikit-learn - Sample Weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html)
