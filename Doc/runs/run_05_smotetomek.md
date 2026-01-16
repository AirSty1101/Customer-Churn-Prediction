# Run #5 - 2025-12-15 (SMOTETomek Resampling)

[← กลับไปหน้าสรุป](../RESULTS.md)

## Configuration

**Key Change:** ใช้ SMOTETomek (Hybrid: Over-sampling + Under-sampling) เพื่อจัดการ class imbalance

- **Logistic Regression:**

  - Preprocessing: `FixedBinnerForLR` + `OneHotEncoder`
  - Features: 25 features (binned + one-hot encoded)
  - **Resampling: SMOTETomek** - SMOTE (over-sampling) + Tomek Links (under-sampling)
  - `class_weight: 'balanced'`
  - `max_iter: 1000`
  - `solver: 'lbfgs'`

- **XGBoost:**

  - Preprocessing: `FixedBinnerForXGBoost` + `Label Encoding`
  - Features: 10 features (binned + label encoded)
  - **Resampling: SMOTETomek** - hybrid approach
  - `n_estimators: 100`
  - `max_depth: 6`
  - `learning_rate: 0.1`
  - `scale_pos_weight: 1.0000` (เพราะ SMOTETomek ทำให้ข้อมูลสมดุลแล้ว)

- **Cross-Validation:** 5-Fold
- **Threshold:** 0.5 (default)

## Imbalance Handling Strategy

**เทคนิคที่ใช้:** SMOTETomek (Hybrid Approach)

**SMOTETomek = SMOTE + Tomek Links:**

1. **SMOTE (Over-sampling):**

   - สร้าง synthetic samples สำหรับ minority class
   - ทำให้ข้อมูลสมดุล

2. **Tomek Links (Under-sampling):**
   - ลบ samples ที่อยู่บน decision boundary (noisy samples)
   - ทำให้ decision boundary ชัดเจนขึ้น

**ข้อดี:**

- ✅ ได้ประโยชน์จากทั้ง over-sampling และ under-sampling
- ✅ ลด noise จาก majority class
- ✅ Dataset ที่ clean กว่า pure SMOTE

**ข้อเสีย:**

- ⚠️ อาจสูญเสียข้อมูลที่มีประโยชน์
- ⚠️ ซับซ้อนกว่า pure over-sampling

**ก่อน SMOTETomek:**

- Class 0 (ไม่ Churn): **5,574** samples
- Class 1 (Churn): **1,426** samples
- **อัตราส่วน**: ~4:1 (ไม่สมดุล)

**หลัง SMOTETomek (XGBoost):**

- Class 0 (ไม่ Churn): **4,227** samples (ลดลงจาก 5,574)
- Class 1 (Churn): **4,227** samples (เพิ่มจาก 1,426)
- **อัตราส่วน**: 1:1 (สมดุลแล้ว! ✅)
- **Total samples**: ลดลงจาก 7,000 → **8,454** samples

**สังเกต:** Tomek Links ลบ majority class samples ที่เป็น noise ออกไป (จาก 5,574 → 4,227)

## Results (Test Set)

| Model               | Accuracy   | Precision  | Recall     | F1         | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.6980     | 0.3708     | 0.6895     | 0.4823     | 0.7600     |
| **XGBoost**         | **0.8033** | **0.5153** | **0.6046** | **0.5564** | **0.8121** |

## Cross-Validation Results

**Logistic Regression:**

- Accuracy: 0.7047 (+/- 0.0069) ✅ Very Stable
- Precision: 0.7090 (+/- 0.0100) ✅ Stable
- Recall: 0.6950 (+/- 0.0163) ✅ Stable
- F1: 0.7017 (+/- 0.0080) ✅ Very Stable
- ROC-AUC: 0.7753 (+/- 0.0047) ✅ Very Stable

**XGBoost:**

- Accuracy: 0.8319 (+/- 0.0672) ⚠️ ผันแปรสูง
- Precision: 0.8373 (+/- 0.0228) ✅ Stable
- Recall: 0.8192 (+/- 0.1430) ❌ ผันแปรสูงมาก (overfitting!)
- F1: 0.8227 (+/- 0.0915) ⚠️ ผันแปรสูง
- ROC-AUC: 0.9084 (+/- 0.0589) ⚠️ ผันแปรสูง

**⚠️ สังเกต:** XGBoost มี CV scores สูงมาก แต่ Test scores ต่ำกว่ามาก → **มี overfitting!**

## Comparison with Run #2, #3, #4

**XGBoost Performance:**

| Metric        | Run #2 (Class Weights) | Run #3 (SMOTE) | Run #4 (ADASYN) | Run #5 (SMOTETomek) | vs Run #2     | vs Best |
| ------------- | ---------------------- | -------------- | --------------- | ------------------- | ------------- | ------- |
| **Accuracy**  | **0.7880**             | 0.8020         | 0.7980          | 0.8033              | **+1.9%** ✅  | #5      |
| **Precision** | **0.4862**             | 0.5123         | 0.5041          | 0.5153              | **+6.0%** ✅  | #5      |
| **Recall**    | **0.6895**             | 0.6144         | 0.6013          | 0.6046              | **-12.3%** ❌ | #2      |
| **F1**        | **0.5703**             | 0.5587         | 0.5484          | 0.5564              | **-2.4%** ❌  | #2      |
| **ROC-AUC**   | **0.8379**             | 0.8170         | 0.8106          | 0.8121              | **-3.1%** ❌  | #2      |

## Observations & Insights

**❌ SMOTETomek ไม่ได้ช่วยปรับปรุง Performance - แย่กว่า Class Weights!**

**1. เปรียบเทียบกับ Run #2 (Class Weights) - Baseline ที่ดีที่สุด:**

- ✅ **Accuracy เพิ่มขึ้นเล็กน้อย** (+1.9%, จาก 78.80% → 80.33%)
- ✅ **Precision เพิ่มขึ้น** (+6.0%, จาก 48.62% → 51.53%) - **ดีที่สุดในทุก Runs!**
- ❌ **Recall ลดลงมาก** (-12.3%, จาก 68.95% → 60.46%) - **นี่คือปัญหาใหญ่!**
- ❌ **F1 ลดลง** (-2.4%, จาก 57.03% → 55.64%)
- ❌ **ROC-AUC ลดลง** (-3.1%, จาก 83.79% → 81.21%)

**2. เปรียบเทียบกับ SMOTE (Run #3) และ ADASYN (Run #4):**

- ✅ **Precision สูงที่สุด** (51.53% vs 51.23% vs 50.41%)
- ⚠️ **Recall ต่ำกว่า SMOTE เล็กน้อย** (60.46% vs 61.44%)
- ⚠️ **ROC-AUC ต่ำกว่า SMOTE เล็กน้อย** (81.21% vs 81.70%)

**🔍 สาเหตุที่ SMOTETomek ไม่ได้ผล:**

1. **Overfitting ชัดเจน:**

   - CV ROC-AUC = 0.9084 (สูงมาก!)
   - Test ROC-AUC = 0.8121 (ต่ำกว่ามาก)
   - **ต่างกัน ~9.6%** → Model overfit กับ training data

2. **Recall Variance สูงมาก:**

   - CV Recall = 0.8192 ± **0.1430** (std สูงมากที่สุด!)
   - แสดงว่า model ไม่ stable มากที่สุดในทุก Runs

3. **Tomek Links อาจลบข้อมูลที่มีประโยชน์:**

   - ลบ majority class จาก 5,574 → 4,227 (ลดลง 24%)
   - อาจลบ samples ที่อยู่ใกล้ decision boundary ที่มีประโยชน์

4. **ยังคงมีปัญหาเดิมกับ Synthetic Sampling:**
   - สร้าง synthetic samples ที่ไม่สมจริง
   - ทำให้ model เรียนรู้ pattern ที่ผิดพลาด

**💡 Key Insights:**

1. **SMOTETomek ดีกว่า SMOTE และ ADASYN เล็กน้อย:**

   - Precision สูงสุด (51.53%)
   - แต่ยังแย่กว่า Class Weights ในทุก metrics ที่สำคัญ

2. **Precision สูงขึ้น แต่ Recall ลดลง:**

   - Trade-off ที่ไม่คุ้มค่า
   - เป้าหมายหลักคือ Recall สูง (จับ Churn ให้ได้มากที่สุด)

3. **Hybrid Approach ไม่ได้ช่วยแก้ปัญหา Overfitting:**

   - ยังคงมี overfitting เหมือน SMOTE และ ADASYN
   - Recall variance สูงที่สุดในทุก Runs

4. **Class Weights (Run #2) ยังคงดีที่สุด:**
   - ROC-AUC = 0.8379 (สูงสุด)
   - Recall = 0.6895 (สูงสุด)
   - ไม่มี overfitting

## Ranking ของ Runs ทั้งหมด

| Rank | Run | Method        | ROC-AUC    | Recall     | F1         | Precision  | Overfitting?   |
| ---- | --- | ------------- | ---------- | ---------- | ---------- | ---------- | -------------- |
| 🥇 1 | #2  | Class Weights | **0.8379** | **0.6895** | **0.5703** | 0.4862     | ❌ No          |
| 🥈 2 | #3  | SMOTE         | 0.8170     | 0.6144     | 0.5587     | 0.5123     | ⚠️ Yes         |
| 🥉 3 | #5  | SMOTETomek    | 0.8121     | 0.6046     | 0.5564     | **0.5153** | ⚠️ Yes         |
| 4    | #4  | ADASYN        | 0.8106     | 0.6013     | 0.5484     | 0.5041     | ⚠️ Yes (worst) |

## Conclusion

**❌ ไม่แนะนำให้ใช้ SMOTETomek กับ dataset นี้:**

1. **Recall ลดลงมาก (-12.3%):**

   - เป้าหมายหลักคือจับ Churn ให้ได้มากที่สุด
   - SMOTETomek ทำให้ Recall ลดลงจาก 68.95% → 60.46%

2. **Overfitting รุนแรง:**

   - Recall variance สูงที่สุด (±0.1430)
   - Model ไม่ stable

3. **Tomek Links อาจลบข้อมูลที่มีประโยชน์:**
   - ลบ 24% ของ majority class
   - อาจลบ samples ที่สำคัญ

**✅ สรุปจากการทดสอบ Synthetic Sampling ทั้งหมด:**

| Method            | ROC-AUC    | Recall     | Precision  | Ranking     |
| ----------------- | ---------- | ---------- | ---------- | ----------- |
| **Class Weights** | **0.8379** | **0.6895** | 0.4862     | 🥇 **Best** |
| SMOTE             | 0.8170     | 0.6144     | 0.5123     | 🥈 2nd      |
| SMOTETomek        | 0.8121     | 0.6046     | **0.5153** | 🥉 3rd      |
| ADASYN            | 0.8106     | 0.6013     | 0.5041     | 4th         |

**💡 ข้อสรุปสำคัญ:**

1. **Class Weights ดีที่สุดในทุกด้าน**
2. **Synthetic Sampling ทุกวิธีสร้าง overfitting**
3. **Hybrid approach ไม่ได้ช่วยแก้ปัญหา**
4. **Precision สูงขึ้น แต่ Recall ลดลง (trade-off ที่ไม่คุ้มค่า)**

**✅ แนะนำให้ใช้ Run #2 (Class Weights) เป็น Final Model**

## Plots

**All visualizations saved in:** `plots/run_5/`

- ✅ `confusion_matrix_lr.png` - Confusion Matrix (Logistic Regression)
- ✅ `confusion_matrix_xgb.png` - Confusion Matrix (XGBoost)
- ✅ `roc_curves.png` - ROC Curves
- ✅ `precision_recall_curves.png` - Precision-Recall Curves
- ✅ `feature_importance_lr.png` - Feature Importance (Logistic Regression)
- ✅ `feature_importance_xgb.png` - Feature Importance (XGBoost)
- ✅ `shap_summary.png` - SHAP Summary Plot
- ✅ `shap_bar.png` - SHAP Feature Importance
- ✅ `shap_waterfall_sample0.png` - SHAP Waterfall (Sample 0)
- ✅ `shap_waterfall_churn.png` - SHAP Waterfall (Churned Customer)
- ✅ `shap_dependence_top.png` - SHAP Dependence Plot (NumOfProducts)

---

[← กลับไปหน้าสรุป](../RESULTS.md)
