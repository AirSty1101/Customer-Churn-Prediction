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
RUN_NUMBER = 1

# Cross-Validation
CV_FOLDS = 5

# Logistic Regression
LR_MAX_ITER = 1000
LR_SOLVER = 'lbfgs'

# XGBoost
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_RANDOM_STATE = 42
