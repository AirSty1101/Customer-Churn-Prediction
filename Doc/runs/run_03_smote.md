# Run #3 - 2025-12-14 (SMOTE Resampling)

[← กลับไปหน้าสรุป](../RESULTS.md)

## Configuration

**Key Change:** ใช้ SMOTE (Synthetic Minority Over-sampling Technique) เพื่อจัดการ class imbalance

- **Logistic Regression:**

  - Preprocessing: `FixedBinnerForLR` + `OneHotEncoder`
  - Features: 25 features (binned + one-hot encoded)
  - **Resampling: SMOTE** - สร้าง synthetic samples สำหรับ minority class
  - `class_weight: 'balanced'`
  - `max_iter: 1000`
  - `solver: 'lbfgs'`

- **XGBoost:**

  - Preprocessing: `FixedBinnerForXGBoost` + `Label Encoding`
  - Features: 10 features (binned + label encoded)
  - **Resampling: SMOTE** - สร้าง synthetic samples สำหรับ minority class
  - `n_estimators: 100`
  - `max_depth: 6`
  - `learning_rate: 0.1`
  - `scale_pos_weight: 1.0000` (เปลี่ยนจาก 3.9088 เพราะ SMOTE ทำให้ข้อมูลสมดุลแล้ว)

- **Cross-Validation:** 5-Fold
- **Threshold:** 0.5 (default)

## Imbalance Handling Strategy

**เทคนิคที่ใช้:** SMOTE (Synthetic Minority Over-sampling Technique)

**ก่อน SMOTE:**

- Class 0 (ไม่ Churn): **5,574** samples
- Class 1 (Churn): **1,426** samples
- **อัตราส่วน**: ~4:1 (ไม่สมดุล)

**หลัง SMOTE:**

- Class 0 (ไม่ Churn): **5,574** samples (เหมือนเดิม)
- Class 1 (Churn): **5,574** samples (เพิ่มจาก 1,426)
- **อัตราส่วน**: 1:1 (สมดุลแล้ว! ✅)
- **Total samples**: เพิ่มจาก 7,000 → **11,148** samples (+4,148 synthetic samples)

**ทำไมใช้ SMOTE:**

- ✅ สร้าง synthetic samples ที่สมจริง (ไม่ใช่แค่ duplicate)
- ✅ ช่วยให้ model เรียนรู้ minority class ได้ดีขึ้น
- ✅ ลด bias ที่มีต่อ majority class
- ⚠️ แต่ต้องระวัง overfitting และ noise จาก synthetic data

## Results (Test Set)

| Model               | Accuracy   | Precision  | Recall     | F1         | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.6980     | 0.3708     | 0.6895     | 0.4823     | 0.7600     |
| **XGBoost**         | **0.8020** | **0.5123** | **0.6144** | **0.5587** | **0.8170** |

## Cross-Validation Results

**Logistic Regression:**

- Accuracy: 0.7047 (+/- 0.0069) ✅ Very Stable
- Precision: 0.7090 (+/- 0.0100) ✅ Stable
- Recall: 0.6950 (+/- 0.0163) ✅ Stable
- F1: 0.7017 (+/- 0.0080) ✅ Very Stable
- ROC-AUC: 0.7753 (+/- 0.0047) ✅ Very Stable

**XGBoost:**

- Accuracy: 0.8315 (+/- 0.0642) ⚠️ ผันแปรสูง (อาจเกิดจาก SMOTE)
- Precision: 0.8391 (+/- 0.0166) ✅ Stable
- Recall: 0.8176 (+/- 0.1436) ⚠️ ผันแปรสูงมาก
- F1: 0.8220 (+/- 0.0891) ⚠️ ผันแปรสูง
- ROC-AUC: 0.9087 (+/- 0.0535) ⚠️ ผันแปรสูง

**⚠️ สังเกต:** XGBoost มี CV scores สูงมาก แต่ Test scores ต่ำกว่ามาก → **อาจมี overfitting จาก SMOTE**

## Comparison with Run #2

**XGBoost Performance Change:**

| Metric        | Run #2 (No SMOTE) | Run #3 (SMOTE) | Change        |
| ------------- | ----------------- | -------------- | ------------- |
| **Accuracy**  | **0.7880**        | 0.8020         | **+1.8%** ✅  |
| **Precision** | **0.4862**        | **0.5123**     | **+5.4%** ✅  |
| **Recall**    | **0.6895**        | 0.6144         | **-10.9%** ❌ |
| **F1**        | **0.5703**        | 0.5587         | **-2.0%** ⚠️  |
| **ROC-AUC**   | **0.8379**        | 0.8170         | **-2.5%** ❌  |

## Observations & Insights

**❌ SMOTE ไม่ได้ช่วยปรับปรุง Performance:**

1. **XGBoost:**

   - ✅ Precision เพิ่มขึ้นเล็กน้อย (+5.4%)
   - ❌ Recall **ลดลงมาก** (-7.5 percentage points, จาก 68.95% → 61.44%) - นี่คือปัญหาใหญ่!
   - ❌ ROC-AUC ลดลง (-2.5%)
   - ⚠️ **Overfitting ชัดเจน:** CV scores สูงมาก (0.91) แต่ Test scores ต่ำกว่า (0.82)

2. **Logistic Regression:**
   - ❌ ทุก metrics ลดลง หรือ เท่าเดิม
   - ไม่ได้ประโยชน์จาก SMOTE เลย

**🔍 สาเหตุที่ SMOTE ไม่ได้ผล:**

1. **Synthetic Data อาจสร้าง Noise:**

   - SMOTE สร้าง 4,148 samples ใหม่ (เพิ่ม 59%)
   - Synthetic samples อาจไม่สะท้อน pattern จริงของลูกค้าที่ Churn

2. **Original Imbalance Ratio ไม่สูงมาก:**

   - Ratio 4:1 ไม่ถือว่าสูงมากนัก
   - Class weights เพียงพอสำหรับ ratio นี้แล้ว (จาก Run #2)

3. **Overfitting:**
   - Model เรียนรู้ pattern จาก synthetic data มากเกินไป
   - ทำให้ generalize กับ real data ได้แย่ลง

**💡 Key Insights:**

1. **Class Weights ดีกว่า SMOTE สำหรับ dataset นี้**

   - Run #2 (Class Weights) ให้ผลดีกว่า Run #3 (SMOTE) ในทุก metrics
   - ไม่ควรใช้ SMOTE กับ dataset นี้

2. **CV Scores ไม่ได้บอกความจริงเสมอไป:**

   - XGBoost CV: ROC-AUC = 0.91 (ดูดีมาก)
   - XGBoost Test: ROC-AUC = 0.82 (ต่ำกว่ามาก)
   - → แสดงว่า model overfit กับ training data (รวม synthetic data)

3. **Recall ลดลง = ปัญหาใหญ่:**
   - เป้าหมายหลักคือจับ Churn ให้ได้มากที่สุด (Recall สูง)
   - SMOTE ทำให้ Recall ลดลง 7.5 percentage points (จาก 68.95% → 61.44%) → ไม่ตอบโจทย์

## Top Features (SHAP Analysis)

**Top 10 Features by Mean Absolute SHAP Value:**

1. **NumOfProducts** (0.8744) - สำคัญที่สุด!
2. **IsActiveMember** (0.8414)
3. **Balance** (0.6552)
4. **Tenure** (0.5870)
5. **Age** (0.3781)
6. **Gender** (0.2793)
7. **Geography** (0.2690)
8. **EstimatedSalary** (0.2601)
9. **HasCrCard** (0.2545)
10. **CreditScore** (0.2277)

**💡 Insights:**

- **NumOfProducts** กลายเป็น feature สำคัญที่สุด (เปลี่ยนจาก Run #2 ที่ Balance สำคัญที่สุด)
- **IsActiveMember** สำคัญมาก - ลูกค้าที่ไม่ Active มีโอกาส Churn สูง
- **Balance** ยังคงสำคัญ แต่ลดลงจากอันดับ 1 → อันดับ 3

## Conclusion

**❌ ไม่แนะนำให้ใช้ SMOTE กับ dataset นี้:**

- Performance แย่กว่า Class Weights ในทุก metrics ที่สำคัญ
- Recall ลดลงมาก (-10.9%)
- มี overfitting ชัดเจน
- Class Weights (Run #2) ยังคงดีที่สุด

## Plots

**All visualizations saved in:** `plots/run_3/`

- ✅ `feature_importance_lr.png` - Feature Importance (Logistic Regression)
- ✅ `feature_importance_xgb.png` - Feature Importance (XGBoost)
- ✅ `shap_summary.png` - SHAP Summary Plot
- ✅ `shap_bar.png` - SHAP Bar Plot
- ✅ `shap_waterfall_sample0.png` - SHAP Waterfall (Sample 0)
- ✅ `shap_waterfall_churn.png` - SHAP Waterfall (Churned Customer)
- ✅ `shap_dependence_top.png` - SHAP Dependence Plot (NumOfProducts)

---

[← กลับไปหน้าสรุป](../RESULTS.md)
