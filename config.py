from pathlib import Path

# === Path พื้นฐาน ===
DATA_PATH = r"C:\Users\absat\Desktop\Customer Churn Prediction\data\Churn_Modelling.csv"

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
