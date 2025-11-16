from __future__ import annotations
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FixedBinner(BaseEstimator, TransformerMixin):
    """
    Transformer สำหรับทำ fixed binning ในคอลัมน์ CreditScore, Age, Tenure, Balance
    ใช้ก่อนเข้า OneHotEncoder
    """

    def __init__(self):
        # คุณสามารถแก้ช่วง bin ได้เองตามต้องการ
        pass

    def fit(self, X, y=None):
        # ไม่มีการเรียนรู้จากข้อมูล
        return self

    def transform(self, X):
        X = X.copy()

        # === Age (ปี) ===
        # 18–25, 26–35, 36–45, 46–60, 60+
        X["Age_bin"] = pd.cut(
            X["Age"],
            bins=[0, 25, 35, 45, 60, 200],
            labels=["18-25", "26-35", "36-45", "46-60", "60+"],
            right=True,
        )

        # === CreditScore ===
        # Low <580, Mid 580–700, High >700
        X["CreditScore_bin"] = pd.cut(
            X["CreditScore"],
            bins=[0, 580, 700, 1000],
            labels=["Low", "Medium", "High"],
            right=True,
        )

        # === Tenure (ปีที่เป็นลูกค้า) ===
        # New <2, Mid 2–5, Loyal >5
        X["Tenure_bin"] = pd.cut(
            X["Tenure"],
            bins=[-1, 1, 5, 20],
            labels=["New", "Mid", "Loyal"],
            right=True,
        )

        # === Balance (ยอดเงิน) ===
        # ธนาคารทั่วไปแบ่ง low/mid/high แบบนี้
        X["Balance_bin"] = pd.cut(
            X["Balance"],
            bins=[-1, 50000, 150000, 1e9],
            labels=["Low", "Medium", "High"],
            right=True,
        )

        # คืนเฉพาะคอลัมน์ใหม่ (bin) + คอลัมน์อื่น ๆ ที่ไม่ใช่ตัวเลขพวกนี้
        # แต่เราจะลบคอลัมน์ numeric ดั้งเดิม
        X = X.drop(columns=["Age", "CreditScore", "Tenure", "Balance"])

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

    # ใช้ transformer
    binner = FixedBinner()
    result = binner.transform(df)

    print(result)
