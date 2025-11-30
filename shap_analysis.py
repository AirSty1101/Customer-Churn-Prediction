"""
SHAP Analysis Script
วิเคราะห์ XGBoost model ด้วย SHAP สำหรับ explainability
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap
from pathlib import Path

from logger_config import setup_logger
from config import MODELS_DIR, PLOTS_DIR

logger = setup_logger("shap_analysis")


def create_directories():
    """สร้างโฟลเดอร์สำหรับเก็บ plots"""
    plots_dir = Path(PLOTS_DIR)
    plots_dir.mkdir(exist_ok=True)
    logger.info(f"Created directory: {plots_dir}")


def load_model():
    """โหลด XGBoost model และ preprocessor"""
    models_dir = Path(MODELS_DIR)
    
    logger.info("Loading XGBoost model...")
    with open(models_dir / "xgboost.pkl", 'rb') as f:
        xgb_model = pickle.load(f)
    
    with open(models_dir / "preprocessor.pkl", 'rb') as f:
        preprocessor = pickle.load(f)
    
    logger.info("Models loaded successfully")
    return xgb_model, preprocessor


def plot_shap_summary(shap_values, X_test_transformed, feature_names, save_path):
    """สร้าง SHAP Summary Plot"""
    logger.info("Creating SHAP summary plot...")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_transformed, 
                     feature_names=feature_names,
                     show=False, max_display=15)
    plt.title('SHAP Summary Plot - XGBoost', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved SHAP summary plot to {save_path}")


def plot_shap_waterfall(explainer, shap_values, X_test_transformed, feature_names, save_path, sample_idx=0):
    """สร้าง SHAP Waterfall Plot สำหรับ 1 sample"""
    logger.info(f"Creating SHAP waterfall plot for sample {sample_idx}...")
    
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=explainer.expected_value,
            data=X_test_transformed[sample_idx],
            feature_names=feature_names
        ),
        show=False,
        max_display=15
    )
    plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved SHAP waterfall plot to {save_path}")


def plot_shap_bar(shap_values, feature_names, save_path):
    """สร้าง SHAP Bar Plot (mean absolute SHAP values)"""
    logger.info("Creating SHAP bar plot...")
    
    # คำนวณ mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # สร้าง DataFrame และ sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False).head(15)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['mean_abs_shap'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Mean |SHAP value|', fontsize=12)
    plt.title('Top 15 Features by Mean Absolute SHAP Value', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved SHAP bar plot to {save_path}")
    
    # พิมพ์ top features
    logger.info("Top 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")


def plot_shap_dependence(shap_values, X_test_transformed, feature_names, save_path, feature_idx=0):
    """สร้าง SHAP Dependence Plot"""
    logger.info(f"Creating SHAP dependence plot for feature: {feature_names[feature_idx]}...")
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature_idx, shap_values, X_test_transformed,
        feature_names=feature_names,
        show=False
    )
    plt.title(f'SHAP Dependence Plot - {feature_names[feature_idx]}', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved SHAP dependence plot to {save_path}")


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("STARTING SHAP ANALYSIS")
    logger.info("="*70)
    
    # สร้างโฟลเดอร์
    create_directories()
    
    # โหลด model
    xgb_model, preprocessor = load_model()
    
    # โหลดข้อมูล test
    from data_prep import get_prepared_data
    logger.info("Loading test data...")
    X_train, X_val, X_test, y_train, y_val, y_test, _ = get_prepared_data()
    
    # Transform
    X_test_transformed = preprocessor.transform(X_test)
    
    # แปลง sparse matrix เป็น dense array (SHAP ต้องการ)
    if hasattr(X_test_transformed, 'toarray'):
        X_test_dense = X_test_transformed.toarray()
    else:
        X_test_dense = X_test_transformed
    
    logger.info(f"Test data shape: {X_test_dense.shape}")
    
    # ดึง feature names
    feature_names = preprocessor.named_steps['encoder'].get_feature_names_out()
    logger.info(f"Number of features: {len(feature_names)}")
    
    # สร้าง SHAP explainer
    logger.info("Creating SHAP explainer (this may take a while)...")
    explainer = shap.TreeExplainer(xgb_model)
    
    # คำนวณ SHAP values
    logger.info("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_test_dense)
    logger.info(f"SHAP values shape: {shap_values.shape}")
    
    plots_dir = Path(PLOTS_DIR)
    
    # 1. Summary Plot
    plot_shap_summary(shap_values, X_test_dense, feature_names,
                     plots_dir / "shap_summary.png")
    
    # 2. Bar Plot
    plot_shap_bar(shap_values, feature_names,
                 plots_dir / "shap_bar.png")
    
    # 3. Waterfall Plot (sample 0 - ลูกค้าคนแรก)
    plot_shap_waterfall(explainer, shap_values, X_test_dense, feature_names,
                       plots_dir / "shap_waterfall_sample0.png", sample_idx=0)
    
    # 4. Waterfall Plot (sample ที่ทำนายว่า Churn)
    y_pred = xgb_model.predict(X_test_dense)
    churn_indices = np.where(y_pred == 1)[0]
    if len(churn_indices) > 0:
        churn_idx = churn_indices[0]
        plot_shap_waterfall(explainer, shap_values, X_test_dense, feature_names,
                           plots_dir / "shap_waterfall_churn.png", sample_idx=churn_idx)
        logger.info(f"Created waterfall plot for churned customer (index {churn_idx})")
    
    # 5. Dependence Plot (feature ที่สำคัญที่สุด)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_feature_idx = np.argmax(mean_abs_shap)
    plot_shap_dependence(shap_values, X_test_dense, feature_names,
                        plots_dir / "shap_dependence_top.png", 
                        feature_idx=top_feature_idx)
    
    logger.info("="*70)
    logger.info("SHAP ANALYSIS COMPLETE!")
    logger.info(f"All plots saved to: {plots_dir.absolute()}")
    logger.info("="*70)
