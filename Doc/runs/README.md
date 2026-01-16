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

2. **[run_02_class_weights.md](run_02_class_weights.md)** ⭐ **BEST OVERALL**

   - วันที่: 2025-12-12
   - เทคนิค: Class Weights + Separate Preprocessing
   - ROC-AUC: **0.8379** (สูงสุด!)
   - F1: **0.5703** (สูงสุด!)

3. **[run_03_smote.md](run_03_smote.md)** - SMOTE Resampling

   - วันที่: 2025-12-14
   - เทคนิค: SMOTE
   - ROC-AUC: 0.8170
   - ❌ Overfitting

4. **[run_04_adasyn.md](run_04_adasyn.md)** - ADASYN Resampling

   - วันที่: 2025-12-15
   - เทคนิค: ADASYN
   - ROC-AUC: 0.8106
   - ❌ Overfitting (worst)

5. **[run_05_smotetomek.md](run_05_smotetomek.md)** - SMOTETomek Resampling

   - วันที่: 2025-12-15
   - เทคนิค: SMOTETomek (Hybrid)
   - ROC-AUC: 0.8121
   - Precision: **0.5153** (สูงสุด!)
   - ❌ Overfitting

6. **[run_06_cost_sensitive.md](run_06_cost_sensitive.md)** 🎯 **HIGHEST RECALL**
   - วันที่: 2026-01-11
   - เทคนิค: Cost-Sensitive Learning
   - Recall: **0.9183** (สูงสุด!)
   - ROC-AUC: 0.8220

## 🏆 Ranking

| Rank | Run | Method            | ROC-AUC    | Recall     | F1         | Best For           |
| ---- | --- | ----------------- | ---------- | ---------- | ---------- | ------------------ |
| 🥇 1 | #2  | Class Weights     | **0.8379** | 0.6895     | **0.5703** | **Overall Best**   |
| 🥈 2 | #6  | Cost-Sensitive    | 0.8220     | **0.9183** | 0.4336     | **Highest Recall** |
| 🥉 3 | #3  | SMOTE             | 0.8170     | 0.6144     | 0.5587     | -                  |
| 4    | #5  | SMOTETomek        | 0.8121     | 0.6046     | 0.5564     | Highest Precision  |
| 5    | #4  | ADASYN            | 0.8106     | 0.6013     | 0.5484     | -                  |
| 6    | #1  | Baseline (OneHot) | 0.7279     | 0.6144     | 0.4460     | -                  |

## 💡 Key Insights

- **Run #2 (Class Weights)** ดีที่สุดสำหรับภาพรวม
- **Run #6 (Cost-Sensitive)** ดีที่สุดสำหรับ Recall
- **Synthetic Sampling** (SMOTE, ADASYN, SMOTETomek) ทุกวิธีสร้าง overfitting
- **Label Encoding** เหมาะกับ XGBoost มากกว่า OneHot

## 🔙 กลับไปหน้าสรุป

[← กลับไปหน้าสรุป](../RESULTS.md)
