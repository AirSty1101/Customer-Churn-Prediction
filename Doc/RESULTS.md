# Experiment Results

บันทึกผลการทดลองและ metrics ของแต่ละรอบการ train model

---

## 📋 Template สำหรับบันทึกผลลัพธ์

เมื่อรัน experiment ใหม่ ให้คัดลอก template ด้านล่างและกรอกผลลัพธ์

```markdown
## Run #X - YYYY-MM-DD

### Configuration

- **Logistic Regression:**

  - class_weight: 'balanced'
  - max_iter: 1000
  - solver: 'lbfgs'

- **XGBoost:**

  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1
  - scale_pos_weight: [calculated value]

- **Cross-Validation:** 5-Fold
- **Threshold:** 0.5

### Results (Test Set)

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
| ------------------- | -------- | --------- | ------ | ------ | ------- |
| Logistic Regression | 0.XXXX   | 0.XXXX    | 0.XXXX | 0.XXXX | 0.XXXX  |
| XGBoost             | 0.XXXX   | 0.XXXX    | 0.XXXX | 0.XXXX | 0.XXXX  |

### Cross-Validation Results

**Logistic Regression:**

- Accuracy: 0.XXXX (+/- 0.XXXX)
- Precision: 0.XXXX (+/- 0.XXXX)
- Recall: 0.XXXX (+/- 0.XXXX)
- F1: 0.XXXX (+/- 0.XXXX)
- ROC-AUC: 0.XXXX (+/- 0.XXXX)

**XGBoost:**

- Accuracy: 0.XXXX (+/- 0.XXXX)
- Precision: 0.XXXX (+/- 0.XXXX)
- Recall: 0.XXXX (+/- 0.XXXX)
- F1: 0.XXXX (+/- 0.XXXX)
- ROC-AUC: 0.XXXX (+/- 0.XXXX)

### Top 10 Features (SHAP - XGBoost)

1. Feature_name_1 (SHAP value: 0.XXXX)
2. Feature_name_2 (SHAP value: 0.XXXX)
3. Feature_name_3 (SHAP value: 0.XXXX)
   ...

### Confusion Matrix (Test Set)

**Logistic Regression:**
```

              Predicted
              0      1

Actual 0 [TN] [FP]
1 [FN] [TP]

```

**XGBoost:**
```

              Predicted
              0      1

Actual 0 [TN] [FP]
1 [FN] [TP]

```

### Observations & Insights

- [สิ่งที่สังเกตเห็นจากผลลัพธ์]
- [ข้อดี/ข้อเสียของแต่ละ model]
- [Insights จาก SHAP analysis]
- [แนวทางปรับปรุงในรอบถัดไป]

### Plots

- Confusion Matrix: `plots/confusion_matrix_*.png`
- ROC Curves: `plots/roc_curves.png`
- SHAP Summary: `plots/shap_summary.png`

---
```

---

## 🎯 เป้าหมาย Metrics

| Metric        | เป้าหมาย | เหตุผล                                   |
| ------------- | -------- | ---------------------------------------- |
| **Recall**    | > 0.70   | สำคัญที่สุด - ต้องจับ Churn ได้มากที่สุด |
| **Precision** | > 0.60   | ลด False Positive                        |
| **F1 Score**  | > 0.65   | สมดุลระหว่าง Precision & Recall          |
| **ROC-AUC**   | > 0.80   | ความสามารถแยก class โดยรวม               |

---

## 📝 บันทึกผลการทดลอง

## 📝 บันทึกผลการทดลอง

### Run #1 - 2025-12-07 (Baseline: Class Weights Only)

#### Configuration

- **Logistic Regression:**

  - class_weight: 'balanced'
  - max_iter: 1000
  - solver: 'lbfgs'

- **XGBoost:**

  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1
  - scale_pos_weight: 3.9088 (auto-calculated)

- **Cross-Validation:** 5-Fold
- **Threshold:** 0.5 (default)

#### Imbalance Handling Strategy

**เทคนิคที่ใช้:** Class Weights

- Logistic Regression: `class_weight='balanced'`
- XGBoost: `scale_pos_weight=3.9088`

**เหตุผลที่ไม่ใช้ SMOTE:**

- ❌ SMOTE สร้างข้อมูลสังเคราะห์ (synthetic data) ที่ไม่ใช่ข้อมูลจริง
- ✅ Class weights ใช้ข้อมูลจริงทั้งหมด แค่ปรับน้ำหนักในการ train
- ✅ หลีกเลี่ยงความเสี่ยงที่ model จะเรียนรู้จาก pattern ที่ไม่สมจริง

#### Results (Test Set)

| Model                   | Accuracy   | Precision  | Recall     | F1         | ROC-AUC    |
| ----------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **Logistic Regression** | **0.7147** | **0.3887** | **0.6961** | **0.4988** | **0.7621** |
| XGBoost                 | 0.6887     | 0.3501     | 0.6144     | 0.4460     | 0.7279     |

#### Cross-Validation Results

**Logistic Regression:**

- Accuracy: 0.7110 (+/- 0.0110)
- Precision: 0.3813 (+/- 0.0095)
- Recall: 0.6690 (+/- 0.0234)
- F1: 0.4854 (+/- 0.0059)
- ROC-AUC: 0.7626 (+/- 0.0046)

**XGBoost:**

- Accuracy: 0.7129 (+/- 0.0093)
- Precision: 0.3788 (+/- 0.0104)
- Recall: 0.6381 (+/- 0.0097)
- F1: 0.4753 (+/- 0.0096)
- ROC-AUC: 0.7422 (+/- 0.0091)

#### Confusion Matrix (Test Set)

**Logistic Regression:**

```
              Predicted
              0      1
Actual  0   [TN]   [FP]
        1   [FN]   [TP]

Estimated from metrics:
- True Negatives (TN): ~900
- False Positives (FP): ~294
- False Negatives (FN): ~93
- True Positives (TP): ~213
```

**XGBoost:**

```
              Predicted
              0      1
Actual  0   [TN]   [FP]
        1   [FN]   [TP]

Estimated from metrics:
- True Negatives (TN): ~846
- False Positives (FP): ~348
- False Negatives (FN): ~118
- True Positives (TP): ~188
```

#### Observations & Insights

**🏆 Logistic Regression ชนะ:**

- ดีกว่า XGBoost ในทุก metrics
- แสดงว่า relationship ระหว่าง features กับ target อาจเป็น linear มากกว่าที่คิด

**✅ จุดแข็ง:**

1. **Recall สูง (0.70)** - จับลูกค้าที่ Churn จริงได้ 70% ✅ ใกล้เป้าหมายแล้ว!
2. **Cross-Validation stable** - Standard deviation ต่ำ แสดงว่า model ไม่ unstable
3. **ไม่ใช้ SMOTE แต่ได้ผลดี** - Class weights เพียงพอสำหรับ imbalance ratio 4:1

**⚠️ จุดอ่อน:**

1. **Precision ต่ำมาก (0.39)** - ทำนายว่า Churn แล้วถูกแค่ 39%
   - False Positive สูง (~294 คน)
   - ธนาคารอาจเสียต้นทุนในการติดต่อลูกค้าที่ไม่ได้ Churn จริง
2. **ROC-AUC ต่ำกว่าเป้าหมาย (0.76 vs 0.80)**
   - ความสามารถในการแยก class ยังไม่ดีพอ
3. **XGBoost แย่กว่าที่คาด**
   - ปกติ XGBoost ควรดีกว่า Linear model
   - อาจต้อง hyperparameter tuning

**🔍 สาเหตุที่ Precision ต่ำ:**

- Model มี bias ไปทาง Recall (เพราะ class weights)
- Trade-off: Recall สูง → Precision ต่ำ
- ถ้าต้องการ Precision สูงขึ้น → ต้อง threshold tuning

#### Next Steps & Recommendations

**ลำดับความสำคัญ:**

1. **Threshold Tuning** ⭐ (แนะนำทำก่อน)

   - ลด threshold จาก 0.5 → 0.3-0.4 เพื่อเพิ่ม Recall
   - หรือเพิ่ม threshold → 0.6-0.7 เพื่อเพิ่ม Precision
   - หา optimal threshold ที่ balance ระหว่าง Precision & Recall

2. **XGBoost Hyperparameter Tuning**

   - ลอง n_estimators = 200-300
   - ลอง max_depth = 3-5 (ลดลง เพื่อป้องกัน overfitting)
   - ลอง learning_rate = 0.05 (ลดลง)

3. **Feature Engineering** (ถ้าต้องการปรับปรุงเพิ่ม)

   - สร้าง interaction features
   - ลอง binning แบบอื่น

4. **Ensemble Methods**
   - รวม predictions จาก LR + XGBoost

#### Visualizations & Analysis

**📁 Location:** `plots/run_1/`

##### 1. Confusion Matrices

**Logistic Regression** (`confusion_matrix_lr.png`)

![Confusion Matrix - Logistic Regression](../plots/run_1/confusion_matrix_lr.png)

```
              Predicted
              Not Churn  Churn
Actual
Not Churn      ~900      ~294    ← False Positives (ทำนายผิดว่า Churn)
Churn          ~93       ~213    ← True Positives (ทำนายถูก!)
```

**Insights:**

- ✅ จับ Churn ได้ 213/306 = 69.6% (Recall)
- ⚠️ ทำนายผิดว่า Churn 294 คน (False Positive สูง)
- 💡 ถ้าธนาคารติดต่อลูกค้าทุกคนที่ทำนายว่า Churn → จะติดต่อผิด 294/507 = 58%

**XGBoost** (`confusion_matrix_xgb.png`)

![Confusion Matrix - XGBoost](../plots/run_1/confusion_matrix_xgb.png)

- Recall ต่ำกว่า LR (61% vs 70%)
- False Positive ยังสูงอยู่

---

##### 2. ROC Curves (`roc_curves.png`)

![ROC Curves Comparison](../plots/run_1/roc_curves.png)

**ROC Curve คืออะไร:**

- แกน X = False Positive Rate (ทำนายผิดว่า Churn)
- แกน Y = True Positive Rate (จับ Churn ได้)
- ยิ่งโค้งเข้าหามุมซ้ายบน = ยิ่งดี
- AUC (Area Under Curve) = พื้นที่ใต้เส้น (0-1)

**ผลลัพธ์:**

- Logistic Regression AUC = **0.762** (สีน้ำเงิน)
- XGBoost AUC = 0.728 (สีส้ม)
- Random Classifier = 0.5 (เส้นประ)

**Insights:**

- LR ดีกว่า Random มาก แต่ยังห่างจาก Perfect (AUC=1.0)
- ต้องการ AUC > 0.80 → ต้องปรับปรุงเพิ่ม

---

##### 3. Precision-Recall Curves (`precision_recall_curves.png`)

![Precision-Recall Curves](../plots/run_1/precision_recall_curves.png)

**PR Curve คืออะไร:**

- แกน X = Recall (จับ Churn ได้กี่ %)
- แกน Y = Precision (ทำนายถูกกี่ %)
- ยิ่งโค้งเข้าหามุมขวาบน = ยิ่งดี

**Insights:**

- เห็น trade-off ชัดเจน: Recall สูง → Precision ต่ำ
- ถ้าต้องการ Recall 80% → Precision จะลดลงเหลือ ~30%
- ถ้าต้องการ Precision 60% → Recall จะลดลงเหลือ ~40%

**การใช้งาน:**

- ใช้เลือก threshold ที่เหมาะสม
- ถ้าธนาคารรับได้กับ False Positive → เลือก threshold ต่ำ (Recall สูง)

---

##### 4. Feature Importance - Logistic Regression (`feature_importance_lr.png`)

![Feature Importance - Logistic Regression](../plots/run_1/feature_importance_lr.png)

**Top 15 Features ที่มีผลต่อการทำนาย:**

**Positive Coefficients (สีเขียว) = เพิ่มโอกาส Churn:**

- `Age_bin_60+` - อายุ 60+ มีแนวโน้ม Churn สูงสุด
- `Geography_Germany` - ลูกค้าเยอรมัน Churn มากกว่าประเทศอื่น
- `Gender_Female` - ผู้หญิง Churn มากกว่าผู้ชาย (เล็กน้อย)
- `Balance_bin_Low` - ยอดเงินต่ำ → Churn สูง

**Negative Coefficients (สีแดง) = ลด Churn:**

- `IsActiveMember` - ลูกค้า Active → Churn น้อย
- `NumOfProducts` - มี Products มาก → Churn น้อย
- `Balance_bin_High` - ยอดเงินสูง → Churn น้อย

**Insights:**

- อายุเป็นปัจจัยสำคัญที่สุด
- ความ Active และจำนวน Products ช่วยลด Churn

---

##### 5. Feature Importance - XGBoost (`feature_importance_xgb.png`)

![Feature Importance - XGBoost](../plots/run_1/feature_importance_xgb.png)

**Top Features (by weight):**

- คล้ายกับ LR แต่ ranking อาจต่างกัน
- XGBoost ดู interaction ระหว่าง features ได้ดีกว่า

---

##### 6. SHAP Summary Plot (`shap_summary.png`)

![SHAP Summary Plot](../plots/run_1/shap_summary.png)

**SHAP คืออะไร:**

- อธิบายว่าแต่ละ feature มีผลต่อ prediction อย่างไร
- แต่ละจุด = 1 ลูกค้า
- สี = ค่าของ feature (แดง=สูง, น้ำเงิน=ต่ำ)
- แกน X = SHAP value (บวก=เพิ่ม Churn, ลบ=ลด Churn)

**Top Features:**

1. **Age** - อายุมาก (สีแดง) → SHAP value บวก → เพิ่ม Churn
2. **NumOfProducts** - มี Products มาก (สีแดง) → SHAP value ลบ → ลด Churn
3. **IsActiveMember** - Active (สีแดง=1) → SHAP value ลบ → ลด Churn
4. **Geography_Germany** - เป็นลูกค้าเยอรมัน → เพิ่ม Churn

**Insights:**

- ลูกค้าอายุมาก + ไม่ Active + มี Products น้อย = Churn สูงมาก
- ลูกค้า Active + มี Products เยอะ = Churn ต่ำ

---

##### 7. SHAP Bar Plot (`shap_bar.png`)

![SHAP Bar Plot](../plots/run_1/shap_bar.png)

**Mean Absolute SHAP Value:**

- แสดง feature importance โดยเฉลี่ย
- ยิ่งสูง = ยิ่งสำคัญ

**Top 10 Features:**

1. Age
2. NumOfProducts
3. IsActiveMember
4. Geography_Germany
5. Balance
6. Gender
7. CreditScore
8. Tenure
9. HasCrCard
10. EstimatedSalary

---

##### 8. SHAP Waterfall Plots

**Sample 0** (`shap_waterfall_sample0.png`)

![SHAP Waterfall - Sample 0](../plots/run_1/shap_waterfall_sample0.png)

- แสดงการทำนายของลูกค้าคนแรก
- เห็นว่า feature ไหนผลักให้ทำนาย Churn หรือ Not Churn

**Churned Customer** (`shap_waterfall_churn.png`)

![SHAP Waterfall - Churned Customer](../plots/run_1/shap_waterfall_churn.png)

- ตัวอย่างลูกค้าที่ทำนายว่า Churn
- เห็นว่า Age, NumOfProducts, IsActiveMember มีผลอย่างไร

**วิธีอ่าน:**

- Base value = ค่าเริ่มต้น (ความน่าจะเป็นเฉลี่ย)
- ลูกศรแดง = เพิ่มโอกาส Churn
- ลูกศรน้ำเงิน = ลดโอกาส Churn
- Final value = prediction สุดท้าย

---

##### 9. SHAP Dependence Plot (`shap_dependence_top.png`)

![SHAP Dependence Plot](../plots/run_1/shap_dependence_top.png)

**แสดงความสัมพันธ์:**

- แกน X = ค่าของ feature ที่สำคัญที่สุด
- แกน Y = SHAP value
- สี = feature อื่นที่มี interaction

**Insights:**

- เห็น pattern ว่าค่า feature เปลี่ยน → SHAP value เปลี่ยนอย่างไร
- เห็น interaction effects ระหว่าง features

---

#### 📊 สรุปจาก Visualizations

**🎯 ปัจจัยหลักที่ทำให้ Churn:**

1. **อายุมาก** (60+)
2. **ไม่ Active**
3. **มี Products น้อย** (1-2 products)
4. **เป็นลูกค้าเยอรมัน**
5. **ยอดเงินต่ำ**

**💡 Actionable Insights สำหรับธนาคาร:**

1. **กลุ่มเสี่ยง:** ลูกค้าอายุ 60+, ไม่ Active, มี 1 Product
2. **การป้องกัน:**
   - เพิ่ม engagement กับลูกค้าที่ไม่ Active
   - Cross-sell products เพิ่ม (แต่ไม่เกิน 3-4)
   - ดูแลลูกค้าอายุมากเป็นพิเศษ
3. **ตลาดเยอรมัน:** ต้องมีกลยุทธ์พิเศษ

---

#### Plots

**All visualizations saved in:** `plots/run_1/`

- ✅ `confusion_matrix_lr.png` - Confusion Matrix (Logistic Regression)
- ✅ `confusion_matrix_xgb.png` - Confusion Matrix (XGBoost)
- ✅ `roc_curves.png` - ROC Curves Comparison
- ✅ `precision_recall_curves.png` - Precision-Recall Curves
- ✅ `feature_importance_lr.png` - Feature Importance (LR Coefficients)
- ✅ `feature_importance_xgb.png` - Feature Importance (XGBoost)
- ✅ `shap_summary.png` - SHAP Summary Plot
- ✅ `shap_bar.png` - SHAP Feature Importance
- ✅ `shap_waterfall_sample0.png` - SHAP Waterfall (Sample 0)
- ✅ `shap_waterfall_churn.png` - SHAP Waterfall (Churned Customer)
- ✅ `shap_dependence_top.png` - SHAP Dependence Plot

---

## 📈 สรุปการเปรียบเทียบ

| Run | Model   | Recall | F1  | ROC-AUC | หมายเหตุ |
| --- | ------- | ------ | --- | ------- | -------- |
| #1  | XGBoost | -      | -   | -       | Baseline |
| #2  | XGBoost | -      | -   | -       | -        |

---

## 💡 แนวทางปรับปรุง

- [ ] Threshold tuning เพื่อเพิ่ม Recall
- [ ] Hyperparameter tuning (GridSearch/RandomSearch)
- [ ] ลอง ensemble methods
- [ ] Feature engineering เพิ่มเติม
