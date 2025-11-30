from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from logger_config import setup_logger

# สร้าง logger สำหรับ module นี้
logger = setup_logger("feature_binning")


class FixedBinner(BaseEstimator, TransformerMixin):
    """
    Transformer สำหรับทำ binning ในคอลัมน์ CreditScore, Age, Tenure, Balance
    - Age, CreditScore, Tenure: Fixed binning (ไม่ต้องเรียนรู้)
    - Balance: Quantile-based binning (เรียนรู้จาก training data)
    
    ใช้ก่อนเข้า OneHotEncoder
    
    Usage:
        binner = FixedBinner()
        binner.fit(X_train)  # เรียนรู้ quantiles จาก training data
        X_train_binned = binner.transform(X_train)
        X_test_binned = binner.transform(X_test)  # ใช้ quantiles เดียวกัน
    
    Raises:
        ValueError: ถ้าคอลัมน์ที่จำเป็นหายไป หรือมีค่านอกช่วง bins
    """

    def __init__(self):
        # คุณสามารถแก้ช่วง bin ได้เองตามต้องการ
        self.required_cols = ["Age", "CreditScore", "Tenure", "Balance"]
        logger.debug("FixedBinner initialized")

    def fit(self, X, y=None):
        logger.debug("FixedBinner.fit() called - learning Balance quantiles from training data")
        X = X.copy()
        
        # เรียนรู้ quantile cutpoints จาก Balance ใน training data
        if "Balance" in X.columns:
            # คำนวณ quantiles (25%, 50%, 75%) และเพิ่ม min, max
            # จะได้ 5 bin edges สำหรับ 4 labels
            quantiles = X["Balance"].quantile([0.25, 0.5, 0.75]).values
            min_val = X["Balance"].min()
            # ใช้ inf เพื่อรองรับค่าที่สูงกว่า max ของ training data
            max_val = np.inf
            
            # สร้าง bins: [min, Q1, Q2, Q3, inf]
            self.balance_quantiles_ = [min_val] + list(quantiles) + [max_val]
            logger.debug(f"Balance quantiles learned: {self.balance_quantiles_}")
        else:
            logger.warning("Balance column not found in training data - quantiles not learned")
            self.balance_quantiles_ = None
            
        return self

    def transform(self, X):
        logger.debug(f"FixedBinner.transform() called with shape {X.shape}")
        X = X.copy()

        # ตรวจสอบคอลัมน์ที่จำเป็น
        missing_cols = [col for col in self.required_cols if col not in X.columns]
        if missing_cols:
            error_msg = f"Missing required columns for binning: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"All required columns present: {self.required_cols}")

        # === Age (ปี) ===
        logger.debug("Binning Age column...")
        X["Age_bin"] = pd.cut(
            X["Age"],
            bins=[0, 20, 30, 40, 50, 60, 200],
            labels=["<20", "20-30", "31-40", "41-50", "51-60", ">60"],
            right=True,
        )
        if X["Age_bin"].isna().any():
            invalid_ages = X.loc[X["Age_bin"].isna(), "Age"].unique()
            error_msg = f"Invalid Age values found (outside bins): {invalid_ages}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(f"Age binning successful. Distribution: {X['Age_bin'].value_counts().to_dict()}")

        # === CreditScore (Risk Grade) ===
        logger.debug("Binning CreditScore column...")
        X["CreditScore_bin"] = pd.cut(
            X["CreditScore"],
            bins=[0, 615, 645, 665, 680, 698, 724, 752, 900],
            labels=["HH", "GG", "FF", "EE", "DD", "CC", "BB", "AA"],
            right=True,
        )
        if X["CreditScore_bin"].isna().any():
            invalid_scores = X.loc[X["CreditScore_bin"].isna(), "CreditScore"].unique()
            error_msg = f"Invalid CreditScore values found (outside bins): {invalid_scores}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(f"CreditScore binning successful. Distribution: {X['CreditScore_bin'].value_counts().to_dict()}")

        # === Tenure (Customer Segment) ===
        logger.debug("Binning Tenure column...")
        X["Tenure_bin"] = pd.cut(
            X["Tenure"],
            bins=[-1, 2, 5, 10, 20],
            labels=["New/At-Risk", "Emerging Loyalty", "Established/Loyal", "Long-Term"],
            right=True,
        )
        if X["Tenure_bin"].isna().any():
            invalid_tenure = X.loc[X["Tenure_bin"].isna(), "Tenure"].unique()
            error_msg = f"Invalid Tenure values found (outside bins): {invalid_tenure}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(f"Tenure binning successful. Distribution: {X['Tenure_bin'].value_counts().to_dict()}")

        # === Balance (Quantile-based) ===
        logger.debug("Binning Balance column using learned quantiles...")
        
        # ตรวจสอบว่ามี quantiles ที่เรียนรู้มาหรือไม่
        if not hasattr(self, 'balance_quantiles_') or self.balance_quantiles_ is None:
            error_msg = "Balance quantiles not learned. Please call fit() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # ลบ duplicate bins และปรับ labels ให้ตรงกัน
        unique_bins = np.unique(self.balance_quantiles_)
        n_bins = len(unique_bins)
        n_labels = n_bins - 1
        
        # สร้าง labels ตามจำนวน bins ที่เหลือ
        if n_labels == 4:
            labels = ["Q1-Low", "Q2-Medium-Low", "Q3-Medium-High", "Q4-High"]
        elif n_labels == 3:
            labels = ["Low", "Medium", "High"]
        elif n_labels == 2:
            labels = ["Low", "High"]
        else:
            labels = [f"Q{i+1}" for i in range(n_labels)]
        
        logger.debug(f"Using {n_bins} unique bins with {n_labels} labels")
        
        # ใช้ pd.cut() กับ bins ที่เรียนรู้มาจาก training data
        X["Balance_bin"] = pd.cut(
            X["Balance"],
            bins=unique_bins,
            labels=labels,
            include_lowest=True
        )
        
        if X["Balance_bin"].isna().any():
            invalid_balance = X.loc[X["Balance_bin"].isna(), "Balance"].unique()
            error_msg = f"Invalid Balance values found (outside bins): {invalid_balance}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(f"Balance binning successful. Distribution: {X['Balance_bin'].value_counts().to_dict()}")

        # ลบคอลัมน์ numeric ดั้งเดิม
        logger.debug(f"Dropping original numeric columns: {self.required_cols}")
        X = X.drop(columns=self.required_cols)
        
        logger.debug(f"FixedBinner.transform() completed. Output shape: {X.shape}")
        return X

if __name__ == "__main__":
    # ทดสอบ transformer แบบง่าย ๆ
    # สร้าง sample ข้อมูล 3 แถว (เอามาจากของคุณ)
    df = pd.DataFrame({
        "CreditScore": [619, 608, 502],
        "Age": [42, 41, 42],
        "Tenure": [2, 1, 8],
        "Balance": [0.00, 83807.86, 159660.80],
        "Geography": ["France", "Spain", "France"],
        "Gender": ["Female", "Female", "Female"]
    })

    # ใช้ transformer (ต้อง fit ก่อน transform)
    binner = FixedBinner()
    binner.fit(df)  # เรียนรู้ quantiles จาก training data
    result = binner.transform(df)  # ใช้ quantiles ที่เรียนรู้มา

    print(result)
