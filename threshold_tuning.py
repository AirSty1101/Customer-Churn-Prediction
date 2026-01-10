"""
Threshold Tuning Script
หา optimal threshold สำหรับ maximize Recall หรือ F1 Score
โดยไม่ต้อง retrain model
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, roc_curve
)

from data_prep import get_prepared_data
from logger_config import setup_logger
from config import RUN_NUMBER, MODELS_DIR

logger = setup_logger("threshold_tuning")


def calculate_metrics_at_threshold(y_true, y_pred_proba, threshold):
    """
    คำนวณ metrics ที่ threshold ที่กำหนด
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Threshold value (0-1)
    
    Returns:
        dict: Metrics at the given threshold
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }


def find_optimal_thresholds(y_true, y_pred_proba, thresholds=None):
    """
    หา optimal thresholds สำหรับ metrics ต่างๆ
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        thresholds: List of thresholds to test (default: 0.1 to 0.9 step 0.05)
    
    Returns:
        DataFrame: Results for all thresholds
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    results = []
    for threshold in thresholds:
        metrics = calculate_metrics_at_threshold(y_true, y_pred_proba, threshold)
        results.append(metrics)
    
    return pd.DataFrame(results)


def plot_threshold_analysis(results_df, output_dir):
    """
    สร้างกราฟแสดงผลของ threshold ต่าง metrics
    
    Args:
        results_df: DataFrame from find_optimal_thresholds
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Threshold Tuning Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: All metrics vs threshold
    ax1 = axes[0, 0]
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        ax1.plot(results_df['threshold'], results_df[metric], 
                marker='o', label=metric.upper(), linewidth=2)
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('All Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Precision vs Recall
    ax2 = axes[0, 1]
    ax2.plot(results_df['recall'], results_df['precision'], 
            marker='o', linewidth=2, color='purple')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Mark some key points
    # Default threshold (0.5)
    default_idx = results_df['threshold'].sub(0.5).abs().idxmin()
    ax2.plot(results_df.loc[default_idx, 'recall'], 
            results_df.loc[default_idx, 'precision'],
            'r*', markersize=15, label='Default (0.5)')
    
    # Best F1
    best_f1_idx = results_df['f1'].idxmax()
    ax2.plot(results_df.loc[best_f1_idx, 'recall'],
            results_df.loc[best_f1_idx, 'precision'],
            'g*', markersize=15, label=f"Best F1 ({results_df.loc[best_f1_idx, 'threshold']:.2f})")
    
    ax2.legend()
    
    # Plot 3: F1 Score vs Threshold
    ax3 = axes[1, 0]
    ax3.plot(results_df['threshold'], results_df['f1'], 
            marker='o', linewidth=2, color='green')
    ax3.axvline(x=results_df.loc[best_f1_idx, 'threshold'], 
               color='red', linestyle='--', label=f"Best F1 at {results_df.loc[best_f1_idx, 'threshold']:.2f}")
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('F1 Score', fontsize=12)
    ax3.set_title('F1 Score vs Threshold', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Recall vs Threshold
    ax4 = axes[1, 1]
    ax4.plot(results_df['threshold'], results_df['recall'], 
            marker='o', linewidth=2, color='blue')
    
    # Mark recall = 0.70 target
    recall_70_idx = results_df['recall'].sub(0.70).abs().idxmin()
    ax4.axhline(y=0.70, color='red', linestyle='--', label='Target Recall (70%)')
    ax4.axvline(x=results_df.loc[recall_70_idx, 'threshold'],
               color='orange', linestyle='--', 
               label=f"Threshold for ~70% Recall ({results_df.loc[recall_70_idx, 'threshold']:.2f})")
    ax4.set_xlabel('Threshold', fontsize=12)
    ax4.set_ylabel('Recall', fontsize=12)
    ax4.set_title('Recall vs Threshold', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "threshold_tuning_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to: {plot_path}")
    
    plt.close()


def main():
    logger.info("="*70)
    logger.info("THRESHOLD TUNING ANALYSIS")
    logger.info("="*70)
    
    # Load model and data
    run_dir = f"run_{RUN_NUMBER}"
    models_dir = Path(MODELS_DIR) / run_dir
    
    logger.info(f"Loading models from: {models_dir}")
    
    # Load XGBoost model (ใช้ XGBoost เพราะดีที่สุด)
    with open(models_dir / "xgboost.pkl", 'rb') as f:
        xgb_model = pickle.load(f)
    
    with open(models_dir / "preprocessor_xgb.pkl", 'rb') as f:
        preprocessor_xgb = pickle.load(f)
    
    logger.info("Models loaded successfully")
    
    # Load test data
    logger.info("Loading test data...")
    X_train, X_val, X_test, y_train, y_val, y_test, _, _ = get_prepared_data()
    
    # Preprocess test data
    X_test_processed = preprocessor_xgb.transform(X_test)
    
    # Get predictions (probabilities)
    logger.info("Getting predictions...")
    y_pred_proba = xgb_model.predict_proba(X_test_processed)[:, 1]
    
    # Test different thresholds
    logger.info("Testing different thresholds...")
    thresholds = np.arange(0.1, 1.0, 0.01)  # Test from 0.1 to 0.9, step 0.01
    results_df = find_optimal_thresholds(y_test, y_pred_proba, thresholds)
    
    # Find optimal thresholds
    logger.info("\n" + "="*70)
    logger.info("OPTIMAL THRESHOLDS")
    logger.info("="*70)
    
    # Best F1
    best_f1_idx = results_df['f1'].idxmax()
    logger.info(f"\n🏆 Best F1 Score:")
    logger.info(f"  Threshold: {results_df.loc[best_f1_idx, 'threshold']:.2f}")
    logger.info(f"  F1: {results_df.loc[best_f1_idx, 'f1']:.4f}")
    logger.info(f"  Precision: {results_df.loc[best_f1_idx, 'precision']:.4f}")
    logger.info(f"  Recall: {results_df.loc[best_f1_idx, 'recall']:.4f}")
    logger.info(f"  Accuracy: {results_df.loc[best_f1_idx, 'accuracy']:.4f}")
    
    # Best Recall
    best_recall_idx = results_df['recall'].idxmax()
    logger.info(f"\n🎯 Best Recall:")
    logger.info(f"  Threshold: {results_df.loc[best_recall_idx, 'threshold']:.2f}")
    logger.info(f"  Recall: {results_df.loc[best_recall_idx, 'recall']:.4f}")
    logger.info(f"  Precision: {results_df.loc[best_recall_idx, 'precision']:.4f}")
    logger.info(f"  F1: {results_df.loc[best_recall_idx, 'f1']:.4f}")
    logger.info(f"  Accuracy: {results_df.loc[best_recall_idx, 'accuracy']:.4f}")
    
    # Threshold for Recall >= 70%
    recall_70_mask = results_df['recall'] >= 0.70
    if recall_70_mask.any():
        recall_70_df = results_df[recall_70_mask]
        # เลือก threshold ที่ให้ Precision สูงสุด (ในกลุ่มที่ Recall >= 70%)
        best_70_idx = recall_70_df['precision'].idxmax()
        logger.info(f"\n✅ Threshold for Recall >= 70% (Best Precision):")
        logger.info(f"  Threshold: {results_df.loc[best_70_idx, 'threshold']:.2f}")
        logger.info(f"  Recall: {results_df.loc[best_70_idx, 'recall']:.4f}")
        logger.info(f"  Precision: {results_df.loc[best_70_idx, 'precision']:.4f}")
        logger.info(f"  F1: {results_df.loc[best_70_idx, 'f1']:.4f}")
        logger.info(f"  Accuracy: {results_df.loc[best_70_idx, 'accuracy']:.4f}")
    
    # Default threshold (0.5)
    default_idx = results_df['threshold'].sub(0.5).abs().idxmin()
    logger.info(f"\n📊 Default Threshold (0.5):")
    logger.info(f"  Threshold: {results_df.loc[default_idx, 'threshold']:.2f}")
    logger.info(f"  Recall: {results_df.loc[default_idx, 'recall']:.4f}")
    logger.info(f"  Precision: {results_df.loc[default_idx, 'precision']:.4f}")
    logger.info(f"  F1: {results_df.loc[default_idx, 'f1']:.4f}")
    logger.info(f"  Accuracy: {results_df.loc[default_idx, 'accuracy']:.4f}")
    
    # Save results
    output_dir = Path("experiments") / f"run_{RUN_NUMBER}_threshold_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "threshold_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")
    
    # Create plots
    logger.info("Creating plots...")
    plot_threshold_analysis(results_df, output_dir)
    
    # Print summary table
    logger.info("\n" + "="*70)
    logger.info("SUMMARY TABLE")
    logger.info("="*70)
    
    summary_df = pd.DataFrame({
        'Scenario': ['Default (0.5)', 'Best F1', 'Best Recall', 'Recall >= 70%'],
        'Threshold': [
            results_df.loc[default_idx, 'threshold'],
            results_df.loc[best_f1_idx, 'threshold'],
            results_df.loc[best_recall_idx, 'threshold'],
            results_df.loc[best_70_idx, 'threshold'] if recall_70_mask.any() else 0
        ],
        'Accuracy': [
            results_df.loc[default_idx, 'accuracy'],
            results_df.loc[best_f1_idx, 'accuracy'],
            results_df.loc[best_recall_idx, 'accuracy'],
            results_df.loc[best_70_idx, 'accuracy'] if recall_70_mask.any() else 0
        ],
        'Precision': [
            results_df.loc[default_idx, 'precision'],
            results_df.loc[best_f1_idx, 'precision'],
            results_df.loc[best_recall_idx, 'precision'],
            results_df.loc[best_70_idx, 'precision'] if recall_70_mask.any() else 0
        ],
        'Recall': [
            results_df.loc[default_idx, 'recall'],
            results_df.loc[best_f1_idx, 'recall'],
            results_df.loc[best_recall_idx, 'recall'],
            results_df.loc[best_70_idx, 'recall'] if recall_70_mask.any() else 0
        ],
        'F1': [
            results_df.loc[default_idx, 'f1'],
            results_df.loc[best_f1_idx, 'f1'],
            results_df.loc[best_recall_idx, 'f1'],
            results_df.loc[best_70_idx, 'f1'] if recall_70_mask.any() else 0
        ]
    })
    
    print("\n" + summary_df.to_string(index=False))
    
    # Save summary
    summary_path = output_dir / "threshold_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSummary saved to: {summary_path}")
    
    logger.info("\n" + "="*70)
    logger.info("THRESHOLD TUNING COMPLETE!")
    logger.info("="*70)
    
    return results_df, summary_df


if __name__ == "__main__":
    results_df, summary_df = main()
