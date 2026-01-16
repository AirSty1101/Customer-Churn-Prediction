from pathlib import Path

# === Path พื้นฐาน ===
DATA_PATH = r"C:\Users\absat\Desktop\Side Project\Customer Churn Prediction\data\Churn_Modelling.csv"

# === Target & Feature Config ===
TARGET_COL = "Exited"

# คอลัมน์ที่ไม่ใช้เป็น feature
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

# คอลัมน์เชิงหมวดหมู่ (categorical)
CATEGORICAL_COLS = ["Geography", "Gender"]

# คอลัมน์เชิงตัวเลข (numeric)
NUMERIC_COLS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

# === Train / Val / Test Split ===
TEST_SIZE = 0.15     # 15% สำหรับ test
VAL_SIZE = 0.15      # 15% สำหรับ val (จากส่วน train ที่เหลือ)
RANDOM_STATE = 42

# === Model Training Config ===
MODELS_DIR = "models"  # โฟลเดอร์เก็บ trained models
PLOTS_DIR = "plots"    # โฟลเดอร์เก็บ visualizations

# Run Number (เปลี่ยนทุกครั้งที่รัน experiment ใหม่)
RUN_NUMBER = 2.2 

# === Imbalanced Data Handling ===
# Options: 'none', 'smote', 'adasyn', 'smote_tomek', 'smote_enn'
RESAMPLING_METHOD = 'none'  # Run #2: 'none' (class weights only)
                             # Run #3: 'smote'
                             # Run #4: 'adasyn'
                             # Run #5: 'smote_tomek'
                             # Run #6: 'none' (cost-sensitive learning)

# === Cost-Sensitive Learning ===
# ใช้ sample_weight แทน resampling
# False Negative (พลาด Churn) มี cost สูงกว่า False Positive
USE_COST_SENSITIVE = False  # Run #6: True (ทดสอบ cost-sensitive)
COST_RATIO = 10.0  # น้ำหนักของ minority class (Churn) เทียบกับ majority class
                   # ค่าที่แนะนำ: 5.0, 10.0, 15.0, 20.0
                   # ยิ่งสูง = ยิ่ง focus ที่ Recall (ลด False Negative)
                             
# === Prediction Threshold ===
# Threshold สำหรับการทำนาย (default = 0.5)
PREDICTION_THRESHOLD = 0.54 
                              
# Cross-Validation
CV_FOLDS = 5

# Logistic Regression
LR_MAX_ITER = 1000
LR_SOLVER = 'lbfgs'

# XGBoost
XGB_N_ESTIMATORS = 50
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.1
XGB_RANDOM_STATE = 42
XGB_SUBSAMPLE = 0.6
XGB_REG_LAMBDA = 0.1
XGB_REG_ALPHA = 0.5
XGB_MIN_CHILD_WEIGHT = 5
XGB_GAMMA = 0.2
XGB_COLSAMPLE_BYTREE = 0.8


