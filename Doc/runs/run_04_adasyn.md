# Run #4 - 2025-12-15 (ADASYN Resampling)

[← กลับไปหน้าสรุป](../RESULTS.md)

## Configuration

**Key Change:** ใช้ ADASYN (Adaptive Synthetic Sampling) เพื่อจัดการ class imbalance

- **Logistic Regression:**

  - Preprocessing: `FixedBinnerForLR` + `OneHotEncoder`
  - Features: 25 features (binned + one-hot encoded)
  - **Resampling: ADASYN** - สร้าง synthetic samples โดยเน้นที่ samples ที่ยากเรียนรู้
  - `class_weight: 'balanced'`
  - `max_iter: 1000`
  - `solver: 'lbfgs'`

- **XGBoost:**

  - Preprocessing: `FixedBinnerForXGBoost` + `Label Encoding`
  - Features: 10 features (binned + label encoded)
  - **Resampling: ADASYN**
  - `n_estimators: 100`
  - `max_depth: 6`
  - `learning_rate: 0.1`
  - `scale_pos_weight: 1.0000`

- **Cross-Validation:** 5-Fold
- **Threshold:** 0.5 (default)

## Imbalance Handling Strategy

**เทคนิคที่ใช้:** ADASYN (Adaptive Synthetic Sampling)

**ADASYN vs SMOTE:**

- **SMOTE**: สร้าง synthetic samples เท่าๆ กันทุก sample
- **ADASYN**: สร้าง synthetic samples **มากกว่า** สำหรับ samples ที่ยากเรียนรู้ (อยู่ใกล้ decision boundary)

**ข้อดี:**

- ✅ Focus ที่ samples ที่ยากเรียนรู้
- ✅ ช่วยให้ model เรียนรู้ edge cases ได้ดีขึ้น

**ข้อเสีย:**

- ⚠️ อาจสร้าง noise มากกว่า SMOTE
- ⚠️ อาจทำให้ overfitting มากกว่า

**ก่อน ADASYN:**

- Class 0 (ไม่ Churn): **5,574** samples
- Class 1 (Churn): **1,426** samples
- **อัตราส่วน**: ~4:1

**หลัง ADASYN:**

- Class 0 (ไม่ Churn): **5,574** samples (เหมือนเดิม)
- Class 1 (Churn): **5,574** samples (เพิ่มจาก 1,426)
- **อัตราส่วน**: 1:1 (สมดุลแล้ว! ✅)

## Results (Test Set)

| Model               | Accuracy   | Precision  | Recall     | F1         | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.6927     | 0.3697     | 0.7190     | 0.4883     | 0.7617     |
| **XGBoost**         | **0.7980** | **0.5041** | **0.6013** | **0.5484** | **0.8106** |

## Cross-Validation Results

**Logistic Regression:**

- Accuracy: 0.7043 (+/- 0.0058) ✅ Very Stable
- Precision: 0.7089 (+/- 0.0097) ✅ Stable
- Recall: 0.6936 (+/- 0.0135) ✅ Stable
- F1: 0.7010 (+/- 0.0069) ✅ Very Stable
- ROC-AUC: 0.7748 (+/- 0.0041) ✅ Very Stable

**XGBoost:**

- Accuracy: 0.8309 (+/- 0.0686) ⚠️ ผันแปรสูง
- Precision: 0.8382 (+/- 0.0175) ✅ Stable
- Recall: 0.8161 (+/- 0.1363) ❌ ผันแปรสูงมาก (worst!)
- F1: 0.8209 (+/- 0.0891) ⚠️ ผันแปรสูง
- ROC-AUC: 0.9078 (+/- 0.0567) ⚠️ ผันแปรสูง

**⚠️ สังเกต:** XGBoost มี overfitting รุนแรงกว่า SMOTE (Recall variance สูงสุด!)

## Comparison with Run #2 and #3

**XGBoost Performance:**

| Metric        | Run #2 (Class Weights) | Run #3 (SMOTE) | Run #4 (ADASYN) | vs Run #2     | vs SMOTE     |
| ------------- | ---------------------- | -------------- | --------------- | ------------- | ------------ |
| **Accuracy**  | **0.7880**             | 0.8020         | 0.7980          | **+1.3%** ✅  | **-0.5%** ⚠️ |
| **Precision** | **0.4862**             | 0.5123         | 0.5041          | **+3.7%** ✅  | **-1.6%** ❌ |
| **Recall**    | **0.6895**             | 0.6144         | 0.6013          | **-12.8%** ❌ | **-2.1%** ❌ |
| **F1**        | **0.5703**             | 0.5587         | 0.5484          | **-3.8%** ❌  | **-1.8%** ❌ |
| **ROC-AUC**   | **0.8379**             | 0.8170         | 0.8106          | **-3.3%** ❌  | **-0.8%** ❌ |

## Observations & Insights

**❌ ADASYN แย่กว่า SMOTE และ Class Weights:**

1. **เปรียบเทียบกับ Run #2 (Class Weights):**

   - Recall ลดลงมาก (-12.8%, จาก 68.95% → 60.13%)
   - ROC-AUC ลดลง (-3.3%)
   - F1 ลดลง (-3.8%)

2. **เปรียบเทียบกับ Run #3 (SMOTE):**

   - แย่กว่า SMOTE ในทุก metrics
   - Recall ลดลงเพิ่มอีก (-2.1%)
   - ROC-AUC ลดลงเพิ่มอีก (-0.8%)

3. **Overfitting รุนแรงที่สุด:**
   - Recall variance = ±0.1363 (สูงที่สุดในทุก Runs!)
   - CV ROC-AUC = 0.91, Test = 0.81 (ต่างกัน 10%)

**🔍 สาเหตุที่ ADASYN แย่กว่า SMOTE:**

1. **Focus ที่ hard samples อาจสร้าง noise มากเกินไป**
2. **Model เรียนรู้ edge cases ที่ไม่สมจริง**
3. **Overfitting รุนแรงกว่า SMOTE**

**💡 Key Insights:**

- **ADASYN ไม่เหมาะกับ dataset นี้**
- **Class Weights (Run #2) ยังคงดีที่สุด**
- **Synthetic Sampling ทุกวิธีสร้าง overfitting**

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

- Pattern คล้ายกับ Run #3 (SMOTE) มาก
- **NumOfProducts** และ **IsActiveMember** ยังคงเป็นปัจจัยสำคัญที่สุด

## Conclusion

**❌ ไม่แนะนำให้ใช้ ADASYN กับ dataset นี้:**

1. **Performance แย่กว่า Class Weights ทุก metrics:**

   - Recall ลดลง 12.8% (จาก 68.95% → 60.13%)
   - ROC-AUC ลดลง 3.3% (จาก 83.79% → 81.06%)

2. **Overfitting รุนแรง:**

   - CV scores สูงมาก แต่ Test scores ต่ำกว่ามาก
   - Recall variance สูงมาก (±0.1363)

3. **ADASYN แย่กว่า SMOTE:**
   - ทุก metrics ต่ำกว่า SMOTE
   - ไม่ได้ดีกว่าตามที่คาดหวัง

**✅ แนะนำให้ใช้ Run #2 (Class Weights) เป็น Final Model**

## Plots

**All visualizations saved in:** `plots/run_4/`

- ✅ `feature_importance_lr.png` - Feature Importance (Logistic Regression)
- ✅ `feature_importance_xgb.png` - Feature Importance (XGBoost)
- ✅ `shap_summary.png` - SHAP Summary Plot
- ✅ `shap_bar.png` - SHAP Feature Importance
- ✅ `shap_waterfall_sample0.png` - SHAP Waterfall (Sample 0)
- ✅ `shap_waterfall_churn.png` - SHAP Waterfall (Churned Customer)
- ✅ `shap_dependence_top.png` - SHAP Dependence Plot (NumOfProducts)

---

[← กลับไปหน้าสรุป](../RESULTS.md)
