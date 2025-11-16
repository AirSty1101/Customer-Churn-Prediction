from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from config import (
    DATA_PATH,
    TARGET_COL,
    DROP_COLS,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
)
from feature_binning import FixedBinner


def load_raw_data() -> pd.DataFrame:
    """โหลดข้อมูลดิบจากไฟล์ CSV"""
    df = pd.read_csv(DATA_PATH)
    return df


def train_val_test_split(df: pd.DataFrame):
    """แยก df เป็น train / val / test โดย stratify ตาม TARGET_COL"""

    # ตัดคอลัมน์ที่ไม่ใช้
    df = df.drop(columns=DROP_COLS)

    # แยก X, y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # แบ่ง train+val vs test ก่อน
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # จากส่วน temp แบ่งเป็น train / val ตามสัดส่วนที่เหลือ
    # สมมติอยากได้ val 15% ของทั้งหมด:
    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_preprocess_pipeline():

    # Fixed binner สำหรับ numeric
    binning = Pipeline(
        steps=[
            ("bin", FixedBinner()),
        ]
    )

    # OneHot สำหรับ categorical + binned features
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # ตอนนี้ numeric cols จะถูก bin → กลายเป็น categorical
    # ดังนั้นเราจะรวม categorical เดิม + ใหม่เข้า ColumnTransformer เดียว
    all_categorical = CATEGORICAL_COLS + [
        "Age_bin",
        "CreditScore_bin",
        "Tenure_bin",
        "Balance_bin",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            # ทำ binning ก่อน แล้วส่งผลไปเข้าท่อ onehot
            ("binning", binning, NUMERIC_COLS),
            ("cat", categorical_transformer, all_categorical),
        ],
        remainder="drop",
    )

    return preprocessor


def get_prepared_data():
    """
    ฟังก์ชันหลักสำหรับใช้ในไฟล์ train model:
    - return X_train, X_val, X_test (ยังไม่ transform)
    - y_train, y_val, y_test
    - preprocessor object (ยังไม่ fit)
    """
    df = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)
    preprocessor = build_preprocess_pipeline()
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


def _print_basic_stats(y_train, y_val, y_test):
    """พิมพ์ class distribution ง่ายๆ ไว้เช็คว่าการ split ถูกต้อง"""
    def class_ratio(y):
        return (y.value_counts(normalize=True) * 100).round(2).to_dict()

    print("=== Dataset Shape ===")
    print(f"Train size: {len(y_train)}")
    print(f"Val size:   {len(y_val)}")
    print(f"Test size:  {len(y_test)}")
    print()

    print("=== Class Distribution (Exited %) ===")
    print(f"Train: {class_ratio(y_train)}")
    print(f"Val:   {class_ratio(y_val)}")
    print(f"Test:  {class_ratio(y_test)}")


if __name__ == "__main__":  
    df_raw = load_raw_data()
    print(f"Raw data shape: {df_raw.shape}")
    print(df_raw.head(3))
    print()

    X_train, X_val, X_test, y_train, y_val, y_test, _ = get_prepared_data()
    _print_basic_stats(y_train, y_val, y_test)
