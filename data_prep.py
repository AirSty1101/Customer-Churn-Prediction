from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path

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
from logger_config import setup_logger

# สร้าง logger สำหรับ module นี้
logger = setup_logger("data_prep")


def load_raw_data() -> pd.DataFrame:
    """
    โหลดข้อมูลดิบจากไฟล์ CSV
    
    Returns:
        pd.DataFrame: ข้อมูลดิบ
    
    Raises:
        FileNotFoundError: ถ้าไฟล์ไม่พบ
        ValueError: ถ้าไฟล์เสียหายหรือคอลัมน์ไม่ครบ
    """
    logger.debug(f"Entering load_raw_data()")
    logger.info(f"Loading data from: {DATA_PATH}")
    
    # ตรวจสอบว่าไฟล์มีอยู่จริง
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        error_msg = f"Data file not found: {DATA_PATH}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.debug(f"File exists. Size: {data_path.stat().st_size / 1024:.2f} KB")
    
    # โหลด CSV
    try:
        df = pd.read_csv(DATA_PATH)
    except pd.errors.EmptyDataError:
        error_msg = "CSV file is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error reading CSV file: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.debug(f"Columns: {list(df.columns)}")
    logger.debug(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # ตรวจสอบคอลัมน์ที่จำเป็น
    required_cols = DROP_COLS + CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns in CSV: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.debug("All required columns present")
    logger.debug(f"Target column '{TARGET_COL}' distribution: {df[TARGET_COL].value_counts().to_dict()}")
    
    return df


def train_val_test_split(df: pd.DataFrame):
    """
    แยก df เป็น train / val / test โดย stratify ตาม TARGET_COL
    
    Args:
        df: DataFrame ที่โหลดมาจาก load_raw_data()
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.debug(f"Entering train_val_test_split() with df shape {df.shape}")
    logger.info("Splitting data into train/val/test sets...")

    # ตัดคอลัมน์ที่ไม่ใช้
    logger.debug(f"Dropping columns: {DROP_COLS}")
    df = df.drop(columns=DROP_COLS)
    logger.debug(f"Shape after dropping: {df.shape}")

    # แยก X, y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    logger.debug(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.debug(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")

    # แบ่ง train+val vs test ก่อน
    logger.debug(f"Splitting train+val vs test (test_size={TEST_SIZE})")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.debug(f"Test set: {len(X_test)} samples")

    # จากส่วน temp แบ่งเป็น train / val ตามสัดส่วนที่เหลือ
    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    logger.debug(f"Splitting train vs val (adjusted val_size={val_size_adjusted:.4f})")

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    logger.info(f"Split complete - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.debug(f"Train target dist: {y_train.value_counts(normalize=True).to_dict()}")
    logger.debug(f"Val target dist: {y_val.value_counts(normalize=True).to_dict()}")
    logger.debug(f"Test target dist: {y_test.value_counts(normalize=True).to_dict()}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_preprocess_pipeline():
    """
    สร้าง preprocessing pipeline ที่ทำงาน 2 ขั้นตอน:
    1. Binning: แปลง numeric features เป็น categorical bins
    2. OneHot Encoding: encode ทั้ง categorical เดิม + binned features
    
    Returns:
        Pipeline: sklearn pipeline object
    """
    logger.debug("Entering build_preprocess_pipeline()")
    logger.info("Building preprocessing pipeline...")
    
    # คอลัมน์ binned ที่จะถูกสร้างจาก FixedBinner
    binned_cols = ["Age_bin", "CreditScore_bin", "Tenure_bin", "Balance_bin"]
    logger.debug(f"Binned columns: {binned_cols}")
    
    # รวม categorical ทั้งหมด (เดิม + binned)
    all_categorical = CATEGORICAL_COLS + binned_cols
    logger.debug(f"All categorical columns for encoding: {all_categorical}")
    
    # Pipeline หลัก
    preprocessor = Pipeline(
        steps=[
            # ขั้นที่ 1: ทำ binning ทั้งหมดก่อน
            ("binner", FixedBinner()),
            
            # ขั้นที่ 2: OneHot encode categorical columns
            ("encoder", ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), all_categorical),
                ],
                remainder="drop",  # drop numeric cols ที่เหลือ
            )),
        ]
    )

    logger.info("Pipeline created successfully")
    logger.debug(f"Pipeline steps: {[step[0] for step in preprocessor.steps]}")
    
    return preprocessor


def get_prepared_data():
    """
    ฟังก์ชันหลักสำหรับใช้ในไฟล์ train model:
    - return X_train, X_val, X_test (ยังไม่ transform)
    - y_train, y_val, y_test
    - preprocessor object (ยังไม่ fit)
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, preprocessor)
    """
    logger.info("="*60)
    logger.info("Starting data preparation pipeline")
    logger.info("="*60)
    
    df = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)
    preprocessor = build_preprocess_pipeline()
    
    logger.info("Data preparation complete!")
    logger.info("="*60)
    
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
