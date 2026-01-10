"""
Cost-Sensitive Learning Module
สร้าง sample weights สำหรับ cost-sensitive learning
เพื่อให้ model focus ที่การลด False Negative (พลาด Churn)
"""

import numpy as np
from logger_config import setup_logger

logger = setup_logger("cost_sensitive")


def create_sample_weights(y_train, cost_ratio=10.0):
    """
    สร้าง sample weights สำหรับ cost-sensitive learning
    
    Args:
        y_train: Training labels (0 = Not Churn, 1 = Churn)
        cost_ratio: น้ำหนักของ minority class (Churn) เทียบกับ majority class
                   ค่าที่แนะนำ: 5.0, 10.0, 15.0, 20.0
                   ยิ่งสูง = ยิ่ง focus ที่ Recall (ลด False Negative)
    
    Returns:
        sample_weights: Array ของ weights สำหรับแต่ละ sample
    
    Example:
        >>> y_train = np.array([0, 0, 1, 0, 1])
        >>> weights = create_sample_weights(y_train, cost_ratio=10.0)
        >>> # weights = [1.0, 1.0, 10.0, 1.0, 10.0]
    """
    logger.info(f"Creating sample weights with cost_ratio={cost_ratio}")
    
    # นับจำนวน samples แต่ละ class
    n_samples = len(y_train)
    n_negative = (y_train == 0).sum()  # Not Churn
    n_positive = (y_train == 1).sum()  # Churn
    
    logger.info(f"Total samples: {n_samples}")
    logger.info(f"  Negative (Not Churn): {n_negative} ({n_negative/n_samples*100:.2f}%)")
    logger.info(f"  Positive (Churn): {n_positive} ({n_positive/n_samples*100:.2f}%)")
    
    # สร้าง sample weights
    # Majority class (Not Churn) = weight 1.0
    # Minority class (Churn) = weight = cost_ratio
    sample_weights = np.where(y_train == 1, cost_ratio, 1.0)
    
    logger.info(f"Sample weights created:")
    logger.info(f"  Not Churn (0): weight = 1.0")
    logger.info(f"  Churn (1): weight = {cost_ratio}")
    logger.info(f"  Total weight: {sample_weights.sum():.2f}")
    logger.info(f"  Average weight: {sample_weights.mean():.2f}")
    
    return sample_weights


def create_balanced_sample_weights(y_train):
    """
    สร้าง sample weights แบบ balanced (คล้าย class_weight='balanced')
    
    Args:
        y_train: Training labels
    
    Returns:
        sample_weights: Array ของ weights
    
    Formula:
        weight[i] = n_samples / (n_classes * n_samples_in_class[i])
    
    Example:
        ถ้ามี 1000 samples (800 Not Churn, 200 Churn)
        - Not Churn weight = 1000 / (2 * 800) = 0.625
        - Churn weight = 1000 / (2 * 200) = 2.5
    """
    logger.info("Creating balanced sample weights")
    
    n_samples = len(y_train)
    n_classes = 2  # Binary classification
    
    # นับจำนวน samples แต่ละ class
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    # คำนวณ weight สำหรับแต่ละ class
    weights_dict = {}
    for cls, count in class_counts.items():
        weights_dict[cls] = n_samples / (n_classes * count)
    
    logger.info(f"Class distribution: {class_counts}")
    logger.info(f"Calculated weights: {weights_dict}")
    
    # สร้าง sample weights
    sample_weights = np.array([weights_dict[y] for y in y_train])
    
    logger.info(f"Sample weights created:")
    logger.info(f"  Total weight: {sample_weights.sum():.2f}")
    logger.info(f"  Average weight: {sample_weights.mean():.2f}")
    
    return sample_weights


def get_sample_weights(y_train, method='cost_ratio', cost_ratio=10.0):
    """
    Factory function สำหรับสร้าง sample weights
    
    Args:
        y_train: Training labels
        method: 'cost_ratio' หรือ 'balanced'
        cost_ratio: น้ำหนักของ minority class (ใช้เมื่อ method='cost_ratio')
    
    Returns:
        sample_weights: Array ของ weights
    """
    if method == 'cost_ratio':
        return create_sample_weights(y_train, cost_ratio)
    elif method == 'balanced':
        return create_balanced_sample_weights(y_train)
    else:
        logger.warning(f"Unknown method: {method}. Using 'cost_ratio' instead.")
        return create_sample_weights(y_train, cost_ratio)


if __name__ == "__main__":
    # Test code
    print("Testing cost-sensitive learning...")
    
    # สร้าง sample imbalanced data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        weights=[0.8, 0.2],  # 80:20 imbalance
        random_state=42
    )
    
    print(f"\nOriginal data:")
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Test cost_ratio method
    print("\n" + "="*60)
    print("Testing cost_ratio method (cost_ratio=10.0)")
    print("="*60)
    weights = get_sample_weights(y, method='cost_ratio', cost_ratio=10.0)
    print(f"Sample weights shape: {weights.shape}")
    print(f"Unique weights: {np.unique(weights)}")
    
    # Test balanced method
    print("\n" + "="*60)
    print("Testing balanced method")
    print("="*60)
    weights_balanced = get_sample_weights(y, method='balanced')
    print(f"Sample weights shape: {weights_balanced.shape}")
    print(f"Unique weights: {np.unique(weights_balanced)}")
    
    # Compare
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)
    print(f"Cost ratio (10.0) - Total weight: {weights.sum():.2f}")
    print(f"Balanced - Total weight: {weights_balanced.sum():.2f}")
