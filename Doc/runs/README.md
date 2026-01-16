# Experiment Runs

โฟลเดอร์นี้เก็บรายละเอียดของแต่ละ run ที่ทำการทดลอง

## 📁 โครงสร้าง

แต่ละไฟล์เก็บรายละเอียดของ run หนึ่งๆ รวมถึง:

- Configuration
- Results (Test Set & Cross-Validation)
- Observations & Insights
- Top Features (SHAP Analysis)
- Visualizations
- Conclusions

## 📝 รายการ Runs

1. **[run_01_baseline.md](run_01_baseline.md)** - Baseline (Class Weights Only)

   - วันที่: 2025-12-07
   - เทคนิค: Class Weights
   - ROC-AUC: 0.7279

2. **[run_02_class_weights.md](run_02_class_weights.md)** - Separate Preprocessing

   - วันที่: 2025-12-12
   - เทคนิค: Class Weights + Separate Preprocessing
   - ROC-AUC: 0.8379
   - F1: 0.5703

3. **[run_02.1_hyperparameter_tuned.md](run_02.1_hyperparameter_tuned.md)** 🚀 **HYPERPARAMETER TUNED**

   - วันที่: 2026-01-16
   - เทคนิค: Hyperparameter Tuning
   - ROC-AUC: **0.8461** (สูงสุด!)
   - Recall: **0.7451** (สูงในกลุ่ม non-cost-sensitive)
   - F1: 0.5794

4. **[run_02.2_threshold_tuned.md](run_02.2_threshold_tuned.md)** ⭐ **MOST BALANCED**

   - วันที่: 2026-01-16
   - เทคนิค: Hyperparameter Tuning + Threshold 0.54
   - ROC-AUC: **0.8461**
   - F1: **0.5811** (สูงสุด!)
   - Precision: **0.4954** (สูงสุดในกลุ่ม Recall >= 70%)
   - Recall: **0.7026**
   - **Balance ที่สุด!**

5. **[run_03_smote.md](run_03_smote.md)** - SMOTE Resampling

   - วันที่: 2025-12-14
   - เทคนิค: SMOTE
   - ROC-AUC: 0.8170
   - ❌ Overfitting

6. **[run_04_adasyn.md](run_04_adasyn.md)** - ADASYN Resampling

   - วันที่: 2025-12-15
   - เทคนิค: ADASYN
   - ROC-AUC: 0.8106
   - ❌ Overfitting (worst)

7. **[run_05_smotetomek.md](run_05_smotetomek.md)** - SMOTETomek Resampling

   - วันที่: 2025-12-15
   - เทคนิค: SMOTETomek (Hybrid)
   - ROC-AUC: 0.8121
   - Precision: 0.5153
   - ❌ Overfitting

8. **[run_06_cost_sensitive.md](run_06_cost_sensitive.md)** 🎯 **EXTREME RECALL**
   - วันที่: 2026-01-11
   - เทคนิค: Cost-Sensitive Learning
   - Recall: **0.9183** (สูงสุดมาก!)
   - ROC-AUC: 0.8220

## 🏆 Ranking

| Rank | Run  | Method                  | ROC-AUC    | Recall     | Precision  | F1         | Best For                    |
| ---- | ---- | ----------------------- | ---------- | ---------- | ---------- | ---------- | --------------------------- |
| 🥇 1 | #2.2 | Hyperparameter + T=0.54 | **0.8461** | **0.7026** | **0.4954** | **0.5811** | **Most Balanced** ⭐        |
| 🥈 2 | #2.1 | Hyperparameter Tuned    | **0.8461** | **0.7451** | 0.4740     | 0.5794     | **High Recall (Non-CS)** 🚀 |
| 🥉 3 | #2   | Class Weights           | 0.8379     | 0.6895     | **0.4862** | 0.5703     | Baseline                    |
| 4    | #6   | Cost-Sensitive          | 0.8220     | **0.9183** | 0.2838     | 0.4336     | **Extreme Recall** 🎯       |
| 5    | #3   | SMOTE                   | 0.8170     | 0.6144     | 0.5123     | 0.5587     | -                           |
| 6    | #5   | SMOTETomek              | 0.8121     | 0.6046     | 0.5153     | 0.5564     | -                           |
| 7    | #4   | ADASYN                  | 0.8106     | 0.6013     | 0.5041     | 0.5484     | -                           |
| 8    | #1   | Baseline (OneHot)       | 0.7279     | 0.6144     | 0.3501     | 0.4460     | -                           |

## 💡 Key Insights

- **Run #2.2 (Hyperparameter + Threshold 0.54)** ⭐ **ดีที่สุดสำหรับ Production!**
  - F1 Score สูงสุด (58.11%)
  - Balance ดีที่สุด
  - ROI สูงสุด (5,789%)
  - Recall เกินเป้าหมาย 70%
- **Run #2.1 (Hyperparameter Tuned)** 🚀 **Recall สูงสุดในกลุ่ม Non-Cost-Sensitive!**

  - Recall = 74.51% (สูงที่สุดในกลุ่มที่ไม่ใช่ Cost-Sensitive)
  - ROC-AUC = 84.61% (สูงสุด!)
  - เหมาะกับการจับ Churn ให้ได้มากโดยไม่ต้องยอมรับ False Positive สูงมาก

- **Run #6 (Cost-Sensitive)** 🎯 **Extreme Recall!**

  - Recall = 91.83% (สูงสุดโดยรวม!)
  - แต่ Precision ต่ำมาก (28.38%)
  - เหมาะกับ campaign พิเศษที่ยอมรับ False Positive สูง

- **Synthetic Sampling** (SMOTE, ADASYN, SMOTETomek) ทุกวิธีสร้าง overfitting
- **Hyperparameter Tuning** ให้ผลดีกว่า Threshold Tuning อย่างมาก
- **Label Encoding** เหมาะกับ XGBoost มากกว่า OneHot

## 🔙 กลับไปหน้าสรุป

[← กลับไปหน้าสรุป](../RESULTS.md)
