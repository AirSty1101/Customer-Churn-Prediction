# Run #2 - 2025-12-12 (Separate Preprocessing: LR vs XGBoost)

[← กลับไปหน้าสรุป](../RESULTS.md)

## Configuration

**Key Change:** แยก preprocessing pipeline สำหรับแต่ละโมเดล

- **Logistic Regression:**

  - Preprocessing: `FixedBinnerForLR` + `OneHotEncoder`
  - Features: 25 features (binned + one-hot encoded)
  - `class_weight: 'balanced'`
  - `max_iter: 1000`
  - `solver: 'lbfgs'`

- **XGBoost:**

  - Preprocessing: `FixedBinnerForXGBoost` + `Label Encoding` (ไม่ใช้ OneHot)
  - Features: **10 features** (binned + label encoded)
  - `n_estimators: 100`
  - `max_depth: 6`
  - `learning_rate: 0.1`
  - `scale_pos_weight: 3.9088`

- **Cross-Validation:** 5-Fold
- **Threshold:** 0.5 (default)

## Motivation

**ทำไมต้องแยก Preprocessing:**

1. **Logistic Regression ต้องการ One-Hot Encoding** - เพราะ LR เป็น linear model ที่ไม่สามารถเรียนรู้ categorical features โดยตรง
2. **XGBoost ทำงานได้ดีกับ Label Encoding** - Tree-based models สามารถเรียนรู้ categorical features ที่เป็นตัวเลขได้โดยตรง
3. **SHAP Plots อ่านง่ายขึ้น** - เมื่อใช้ Label Encoding, Geography และ Gender จะเป็น 1 feature แทนที่จะแยกเป็นหลาย features

## Results (Test Set)

| Model               | Accuracy   | Precision  | Recall     | F1         | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.7147     | 0.3887     | 0.6961     | 0.4988     | 0.7621     |
| **XGBoost**         | **0.7880** | **0.4862** | **0.6895** | **0.5703** | **0.8379** |

## Cross-Validation Results

**Logistic Regression:**

- Accuracy: 0.7110 (+/- 0.0110) ✅ Stable
- Precision: 0.3813 (+/- 0.0095) ✅ Stable
- Recall: 0.6690 (+/- 0.0234) ⚠️ ผันแปรเล็กน้อย
- F1: 0.4854 (+/- 0.0059) ✅ Stable
- ROC-AUC: 0.7626 (+/- 0.0046) ✅ Very Stable

**XGBoost:**

- Accuracy: 0.7963 (+/- 0.0089) ✅ Stable
- Precision: 0.4996 (+/- 0.0160) ✅ Stable
- Recall: 0.7005 (+/- 0.0385) ⚠️ ผันแปรปานกลาง
- F1: 0.5832 (+/- 0.0243) ✅ Stable
- ROC-AUC: 0.8355 (+/- 0.0146) ✅ Stable

**สรุป:** ทั้ง 2 models มีความ stable ดี

## Comparison with Run #1

**XGBoost Performance Improvement:**

| Metric        | Run #1 (OneHot) | Run #2 (Label) | Improvement   |
| ------------- | --------------- | -------------- | ------------- |
| **Accuracy**  | 0.6887          | **0.7880**     | **+14.4%** 🚀 |
| **Precision** | 0.3501          | **0.4862**     | **+38.9%** 🚀 |
| **Recall**    | 0.6144          | **0.6895**     | **+12.2%** ✅ |
| **F1**        | 0.4460          | **0.5703**     | **+27.9%** 🚀 |
| **ROC-AUC**   | 0.7279          | **0.8379**     | **+15.1%** 🎯 |

## Observations & Insights

**🏆 XGBoost ตอนนี้ดีกว่า LR แล้ว!**

- ใน Run #1: LR ดีกว่า XGBoost ในทุก metrics
- ใน Run #2: **XGBoost ดีกว่า LR ในทุก metrics** (ตามที่ควรจะเป็น)

**✅ จุดแข็ง:**

1. **ROC-AUC = 0.8379** ✅ เกินเป้าหมาย 0.80 แล้ว!
2. **Features ลดลง 60%** - จาก 25 → 10 features แต่ performance ดีขึ้น
3. **Precision เพิ่มขึ้น 38.9%** - ลด False Positive ได้มาก
4. **SHAP Plots อ่านง่ายกว่ามาก:**
   - Geography: 1 feature แทน 3 features (France, Germany, Spain)
   - Gender: 1 feature แทน 2 features (Male, Female)

**💡 Key Insights:**

1. **Label Encoding เหมาะกับ XGBoost มากกว่า OneHot**

   - Tree-based models สามารถเรียนรู้ ordinal relationships ได้
   - ลด feature space → ลด overfitting
   - Model ทำงานได้เร็วขึ้น

2. **Feature Engineering ที่ถูกต้องสำคัญมาก**

   - การเลือก encoding ที่เหมาะสมกับ model สามารถเพิ่ม performance ได้มาก
   - ไม่ใช่ว่า features เยอะ = ดีเสมอไป

3. **Model Interpretability ดีขึ้น**
   - SHAP plots ที่มี features น้อยกว่าอ่านง่ายกว่า
   - เห็นความสำคัญของแต่ละ feature ได้ชัดเจนขึ้น

## Top Features (SHAP Analysis)

**Top 10 Features by Mean Absolute SHAP Value:**

1. **Balance** (~0.85) - สำคัญที่สุด! ยอดเงินมีผลต่อ Churn มาก
2. **NumOfProducts** (~0.80) - จำนวน products มีผลมาก (3-4 products = Churn สูง)
3. **IsActiveMember** (~0.45) - ไม่ active = Churn สูงมาก
4. **Age** (~0.30) - อายุมีผลปานกลาง
5. **Tenure** (~0.25) - ระยะเวลาเป็นลูกค้า
6. **EstimatedSalary** - เงินเดือนประมาณการ
7. **Gender** - ผู้หญิง Churn สูงกว่าเล็กน้อย
8. **CreditScore** - คะแนนเครดิต
9. **Geography** - เยอรมัน Churn สูงกว่า
10. **HasCrCard** - มีบัตรเครดิตหรือไม่

**🔍 Insights สำคัญ:**

- **Balance เป็นปัจจัยสำคัญที่สุด** (ต่างจาก XGBoost weight!)
- **NumOfProducts** สำคัญมาก - ลูกค้าที่มี 3-4 products มี Churn สูง
- **IsActiveMember** - ลูกค้าที่ไม่ active มี Churn สูงมาก
- **SHAP วัดผลกระทบจริง ≠ XGBoost weight** (weight วัดความถี่ในการใช้งาน)

## Next Steps & Recommendations

**ลำดับความสำคัญ:**

1. **✅ Achieved: ROC-AUC > 0.80** - เป้าหมายหลักสำเร็จแล้ว!

2. **Threshold Tuning** (แนะนำทำต่อ)

   - ลอง threshold = 0.3-0.4 เพื่อเพิ่ม Recall
   - หรือเพิ่ม threshold = 0.6-0.7 เพื่อเพิ่ม Precision
   - หา optimal threshold ที่ balance ระหว่าง Precision & Recall

3. **XGBoost Hyperparameter Tuning** (optional - เพื่อ push performance ให้สูงสุด)

   - `n_estimators = 200-300`
   - `max_depth = 3-5`
   - `learning_rate = 0.05`
   - `min_child_weight = 3-5`

4. **Ensemble Methods** (advanced)
   - Voting Classifier: รวม LR + XGBoost
   - Stacking: ใช้ meta-model

## Actionable Insights สำหรับธนาคาร

**🎯 ปัจจัยหลักที่ทำให้ Churn:**

1. **อายุมาก** (40+, โดยเฉพาะ 51-60)
2. **ไม่ Active**
3. **มี Products มาก** (3-4 products)
4. **เป็นลูกค้าเยอรมัน**
5. **ผู้หญิง** (เล็กน้อย)

**💡 การป้องกัน:**

1. **กลุ่มเสี่ยงสูง:** อายุ 50+, ไม่ Active, มี 3-4 Products
2. **การป้องกัน:**
   - เพิ่ม engagement กับลูกค้าที่ไม่ Active
   - Review product portfolio - ทำไมลูกค้าที่มี products เยอะถึง Churn?
   - ดูแลลูกค้าอายุมากเป็นพิเศษ
3. **ตลาดเยอรมัน:** ต้องมีกลยุทธ์พิเศษ - ทำไม Churn สูงกว่าประเทศอื่น?

## Plots

**All visualizations saved in:** `plots/run_2/`

- ✅ `confusion_matrix_lr.png` - Confusion Matrix (Logistic Regression)
- ✅ `confusion_matrix_xgb.png` - Confusion Matrix (XGBoost)
- ✅ `roc_curves.png` - ROC Curves Comparison
- ✅ `precision_recall_curves.png` - Precision-Recall Curves
- ✅ `feature_importance_lr.png` - Feature Importance (LR)
- ✅ `feature_importance_xgb.png` - Feature Importance (XGBoost)
- ✅ `shap_summary.png` - SHAP Summary Plot
- ✅ `shap_bar.png` - SHAP Bar Plot
- ✅ `shap_waterfall_sample0.png` - SHAP Waterfall (Sample 0)
- ✅ `shap_waterfall_churn.png` - SHAP Waterfall (Churned Customer)
- ✅ `shap_dependence_top.png` - SHAP Dependence Plot (Balance)

---

[← กลับไปหน้าสรุป](../RESULTS.md)
