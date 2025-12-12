"""
Imbalanced Data Handling Techniques
รวม SMOTE, ADASYN, Hybrid methods สำหรับจัดการ imbalanced dataset
"""

import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from logger_config import setup_logger

logger = setup_logger("imbalance_handlers")


def apply_none(X_train, y_train, random_state=42):
    """
    ไม่ใช้ resampling technique ใดๆ - ใช้ class weights แทน
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed (ไม่ได้ใช้)
    
    Returns:
        X_train, y_train (ไม่เปลี่ยนแปลง)
    """
    logger.info("No resampling applied - using class weights only")
    logger.info(f"Original shape: {X_train.shape}")
    
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))
    logger.info(f"Class distribution: {class_dist}")
    
    return X_train, y_train


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique)
    สร้าง synthetic samples สำหรับ minority class
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
    
    Returns:
        X_resampled, y_resampled
    """
    logger.info("Applying SMOTE...")
    logger.info(f"Original shape: {X_train.shape}")
    
    unique, counts = np.unique(y_train, return_counts=True)
    logger.info(f"Original class distribution: {dict(zip(unique, counts))}")
    
    smote = SMOTE(random_state=random_state, sampling_strategy='auto')
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    unique, counts = np.unique(y_resampled, return_counts=True)
    logger.info(f"Resampled shape: {X_resampled.shape}")
    logger.info(f"Resampled class distribution: {dict(zip(unique, counts))}")
    
    return X_resampled, y_resampled


def apply_adasyn(X_train, y_train, random_state=42):
    """
    Apply ADASYN (Adaptive Synthetic Sampling)
    Focus ที่ samples ที่ยากเรียนรู้
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
    
    Returns:
        X_resampled, y_resampled
    """
    logger.info("Applying ADASYN...")
    logger.info(f"Original shape: {X_train.shape}")
    
    unique, counts = np.unique(y_train, return_counts=True)
    logger.info(f"Original class distribution: {dict(zip(unique, counts))}")
    
    adasyn = ADASYN(random_state=random_state, sampling_strategy='auto')
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    
    unique, counts = np.unique(y_resampled, return_counts=True)
    logger.info(f"Resampled shape: {X_resampled.shape}")
    logger.info(f"Resampled class distribution: {dict(zip(unique, counts))}")
    
    return X_resampled, y_resampled


def apply_smote_tomek(X_train, y_train, random_state=42):
    """
    Apply SMOTETomek (SMOTE + Tomek Links)
    Over-sample minority class + Under-sample majority class
    ลด noise และ redundant samples
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
    
    Returns:
        X_resampled, y_resampled
    """
    logger.info("Applying SMOTETomek (Hybrid: Over + Under sampling)...")
    logger.info(f"Original shape: {X_train.shape}")
    
    unique, counts = np.unique(y_train, return_counts=True)
    logger.info(f"Original class distribution: {dict(zip(unique, counts))}")
    
    smote_tomek = SMOTETomek(random_state=random_state)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
    
    unique, counts = np.unique(y_resampled, return_counts=True)
    logger.info(f"Resampled shape: {X_resampled.shape}")
    logger.info(f"Resampled class distribution: {dict(zip(unique, counts))}")
    
    return X_resampled, y_resampled


def apply_smote_enn(X_train, y_train, random_state=42):
    """
    Apply SMOTEENN (SMOTE + Edited Nearest Neighbors)
    Over-sample minority class + Clean noisy samples
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
    
    Returns:
        X_resampled, y_resampled
    """
    logger.info("Applying SMOTEENN (Hybrid: Over + Cleaning)...")
    logger.info(f"Original shape: {X_train.shape}")
    
    unique, counts = np.unique(y_train, return_counts=True)
    logger.info(f"Original class distribution: {dict(zip(unique, counts))}")
    
    smote_enn = SMOTEENN(random_state=random_state)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    
    unique, counts = np.unique(y_resampled, return_counts=True)
    logger.info(f"Resampled shape: {X_resampled.shape}")
    logger.info(f"Resampled class distribution: {dict(zip(unique, counts))}")
    
    return X_resampled, y_resampled


def get_resampler(method='none', random_state=42):
    """
    Factory function to get resampler
    
    Args:
        method: 'none', 'smote', 'adasyn', 'smote_tomek', 'smote_enn'
        random_state: Random seed
    
    Returns:
        Resampler function
    """
    resamplers = {
        'none': apply_none,
        'smote': apply_smote,
        'adasyn': apply_adasyn,
        'smote_tomek': apply_smote_tomek,
        'smote_enn': apply_smote_enn,
    }
    
    method_lower = method.lower() if method else 'none'
    
    if method_lower not in resamplers:
        logger.warning(f"Unknown method: {method}. Using 'none' instead.")
        logger.warning(f"Available methods: {list(resamplers.keys())}")
        method_lower = 'none'
    
    logger.info(f"Selected resampling method: {method_lower}")
    return resamplers[method_lower]


if __name__ == "__main__":
    # Test code
    print("Testing imbalance handlers...")
    
    # Create sample imbalanced data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        weights=[0.8, 0.2],  # 80:20 imbalance
        random_state=42
    )
    
    print(f"\nOriginal data shape: {X.shape}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"Original class distribution: {dict(zip(unique, counts))}")
    
    # Test each method
    methods = ['none', 'smote', 'adasyn', 'smote_tomek', 'smote_enn']
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing: {method}")
        print('='*60)
        
        resampler = get_resampler(method)
        X_res, y_res = resampler(X, y)
        
        print(f"Result shape: {X_res.shape}")
        unique, counts = np.unique(y_res, return_counts=True)
        print(f"Result class distribution: {dict(zip(unique, counts))}")
