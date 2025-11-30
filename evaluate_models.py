"""
Model Evaluation Script
สร้าง visualizations: Confusion Matrix, ROC Curves, Feature Importance
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from logger_config import setup_logger
from config import MODELS_DIR, PLOTS_DIR

logger = setup_logger("evaluate_models")

# ตั้งค่า matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_directories():
    """สร้างโฟลเดอร์สำหรับเก็บ plots"""
    plots_dir = Path(PLOTS_DIR)
    plots_dir.mkdir(exist_ok=True)
    logger.info(f"Created directory: {plots_dir}")


def load_models():
    """โหลด trained models"""
    models_dir = Path(MODELS_DIR)
    
    logger.info("Loading models...")
    with open(models_dir / "logistic_regression.pkl", 'rb') as f:
        lr_model = pickle.load(f)
    
    with open(models_dir / "xgboost.pkl", 'rb') as f:
        xgb_model = pickle.load(f)
    
    with open(models_dir / "preprocessor.pkl", 'rb') as f:
        preprocessor = pickle.load(f)
    
    logger.info("Models loaded successfully")
    return lr_model, xgb_model, preprocessor


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """สร้าง Confusion Matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Churn', 'Churn'],
                yticklabels=['Not Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {save_path}")


def plot_roc_curves(y_test, lr_proba, xgb_proba, save_path):
    """สร้าง ROC Curves สำหรับทั้ง 2 models"""
    # คำนวณ ROC curve
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
    lr_auc = auc(lr_fpr, lr_tpr)
    
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)
    xgb_auc = auc(xgb_fpr, xgb_tpr)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.3f})', linewidth=2)
    plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved ROC curves to {save_path}")


def plot_precision_recall_curves(y_test, lr_proba, xgb_proba, save_path):
    """สร้าง Precision-Recall Curves"""
    # คำนวณ PR curve
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_proba)
    xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_proba)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(lr_recall, lr_precision, label='Logistic Regression', linewidth=2)
    plt.plot(xgb_recall, xgb_precision, label='XGBoost', linewidth=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved PR curves to {save_path}")


def plot_feature_importance_lr(lr_model, feature_names, save_path, top_n=15):
    """สร้าง Feature Importance plot สำหรับ Logistic Regression"""
    # ดึง coefficients
    coefficients = lr_model.coef_[0]
    
    # สร้าง DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['red' if x < 0 else 'green' for x in importance_df['coefficient']]
    plt.barh(range(len(importance_df)), importance_df['coefficient'], color=colors)
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance - Logistic Regression', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved LR feature importance to {save_path}")


def plot_feature_importance_xgb(xgb_model, save_path, top_n=15):
    """สร้าง Feature Importance plot สำหรับ XGBoost"""
    from xgboost import plot_importance
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_importance(xgb_model, ax=ax, max_num_features=top_n, importance_type='weight')
    plt.title(f'Top {top_n} Feature Importance - XGBoost', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved XGBoost feature importance to {save_path}")


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("STARTING MODEL EVALUATION")
    logger.info("="*70)
    
    # สร้างโฟลเดอร์
    create_directories()
    
    # โหลด models
    lr_model, xgb_model, preprocessor = load_models()
    
    # โหลดข้อมูล test (ต้อง import และ transform)
    from data_prep import get_prepared_data
    logger.info("Loading test data...")
    X_train, X_val, X_test, y_train, y_val, y_test, _ = get_prepared_data()
    
    # Transform
    X_test_transformed = preprocessor.transform(X_test)
    
    # Predictions
    logger.info("Generating predictions...")
    lr_pred = lr_model.predict(X_test_transformed)
    lr_proba = lr_model.predict_proba(X_test_transformed)[:, 1]
    
    xgb_pred = xgb_model.predict(X_test_transformed)
    xgb_proba = xgb_model.predict_proba(X_test_transformed)[:, 1]
    
    plots_dir = Path(PLOTS_DIR)
    
    # 1. Confusion Matrices
    logger.info("Creating confusion matrices...")
    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression", 
                         plots_dir / "confusion_matrix_lr.png")
    plot_confusion_matrix(y_test, xgb_pred, "XGBoost", 
                         plots_dir / "confusion_matrix_xgb.png")
    
    # 2. ROC Curves
    logger.info("Creating ROC curves...")
    plot_roc_curves(y_test, lr_proba, xgb_proba, 
                   plots_dir / "roc_curves.png")
    
    # 3. Precision-Recall Curves
    logger.info("Creating Precision-Recall curves...")
    plot_precision_recall_curves(y_test, lr_proba, xgb_proba,
                                plots_dir / "precision_recall_curves.png")
    
    # 4. Feature Importance
    logger.info("Creating feature importance plots...")
    
    # สำหรับ LR ต้องดึง feature names จาก preprocessor
    feature_names = preprocessor.named_steps['encoder'].get_feature_names_out()
    plot_feature_importance_lr(lr_model, feature_names, 
                              plots_dir / "feature_importance_lr.png")
    
    plot_feature_importance_xgb(xgb_model, 
                               plots_dir / "feature_importance_xgb.png")
    
    logger.info("="*70)
    logger.info("EVALUATION COMPLETE!")
    logger.info(f"All plots saved to: {plots_dir.absolute()}")
    logger.info("="*70)
