"""
Cost-Sensitive Learning Experiment Script
ทดสอบ cost ratios ต่างๆ เพื่อหา optimal cost ratio
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import configurations
import sys
sys.path.append(str(Path(__file__).parent))

from config import RUN_NUMBER
from data_prep import get_prepared_data
from imbalance_handlers import get_resampler
from cost_sensitive import get_sample_weights
from train_models import train_xgboost, calculate_metrics
from logger_config import setup_logger

logger = setup_logger("cost_sensitive_experiment")


def test_cost_ratios(cost_ratios=[5.0, 10.0, 15.0, 20.0]):
    """
    ทดสอบ cost ratios ต่างๆ และเปรียบเทียบผลลัพธ์
    
    Args:
        cost_ratios: List of cost ratios to test
    
    Returns:
        results_df: DataFrame containing results for each cost ratio
    """
    logger.info("="*70)
    logger.info("COST-SENSITIVE LEARNING EXPERIMENT")
    logger.info("="*70)
    logger.info(f"Testing cost ratios: {cost_ratios}")
    
    # Load data
    logger.info("Loading and preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor_lr, preprocessor_xgb = get_prepared_data()
    
    # Prepare XGBoost data
    logger.info("Preparing XGBoost data...")
    preprocessor_xgb.fit(X_train, y_train)
    X_train_xgb = preprocessor_xgb.transform(X_train)
    X_val_xgb = preprocessor_xgb.transform(X_val)
    X_test_xgb = preprocessor_xgb.transform(X_test)
    
    # No resampling for this experiment
    resampler = get_resampler('none')
    X_train_xgb_resampled, y_train_xgb_resampled = resampler(X_train_xgb, y_train)
    
    # Store results
    results = []
    
    # Baseline: No cost-sensitive (just class weights via scale_pos_weight)
    logger.info("\n" + "="*70)
    logger.info("BASELINE: No Cost-Sensitive (scale_pos_weight only)")
    logger.info("="*70)
    
    xgb_model, xgb_cv, xgb_val_metrics, xgb_test_metrics = train_xgboost(
        X_train_xgb_resampled, y_train_xgb_resampled,
        X_val_xgb, y_val,
        X_test_xgb, y_test,
        sample_weight=None
    )
    
    results.append({
        'cost_ratio': 0.0,
        'method': 'Baseline (scale_pos_weight)',
        **xgb_test_metrics
    })
    
    # Test each cost ratio
    for cost_ratio in cost_ratios:
        logger.info("\n" + "="*70)
        logger.info(f"TESTING COST RATIO: {cost_ratio}")
        logger.info("="*70)
        
        # Create sample weights
        sample_weights = get_sample_weights(
            y_train_xgb_resampled,
            method='cost_ratio',
            cost_ratio=cost_ratio
        )
        
        # Train model
        xgb_model, xgb_cv, xgb_val_metrics, xgb_test_metrics = train_xgboost(
            X_train_xgb_resampled, y_train_xgb_resampled,
            X_val_xgb, y_val,
            X_test_xgb, y_test,
            sample_weight=sample_weights
        )
        
        # Store results
        results.append({
            'cost_ratio': cost_ratio,
            'method': f'Cost-Sensitive (ratio={cost_ratio})',
            **xgb_test_metrics
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Print comparison
    logger.info("\n" + "="*70)
    logger.info("RESULTS COMPARISON")
    logger.info("="*70)
    print("\n" + results_df.to_string(index=False))
    
    # Find best cost ratio for each metric
    logger.info("\n" + "="*70)
    logger.info("BEST COST RATIO FOR EACH METRIC")
    logger.info("="*70)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        best_idx = results_df[metric].idxmax()
        best_row = results_df.iloc[best_idx]
        logger.info(f"{metric.upper()}: {best_row['method']} = {best_row[metric]:.4f}")
    
    # Save results
    output_dir = Path("experiments") / f"run_{RUN_NUMBER}_cost_sensitive"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "cost_ratio_comparison.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")
    
    return results_df


def plot_cost_ratio_comparison(results_df):
    """
    สร้างกราฟเปรียบเทียบ cost ratios
    
    Args:
        results_df: DataFrame containing results
    """
    import matplotlib.pyplot as plt
    
    logger.info("Creating comparison plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Cost-Sensitive Learning: Cost Ratio Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Plot
        x = results_df['cost_ratio']
        y = results_df[metric]
        
        ax.plot(x, y, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Cost Ratio', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} vs Cost Ratio', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Mark best value
        best_idx = y.idxmax()
        ax.plot(x.iloc[best_idx], y.iloc[best_idx], 'r*', markersize=15, label='Best')
        ax.legend()
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("experiments") / f"run_{RUN_NUMBER}_cost_sensitive"
    plot_path = output_dir / "cost_ratio_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to: {plot_path}")
    
    plt.close()


if __name__ == "__main__":
    logger.info("Starting Cost-Sensitive Learning Experiment...")
    
    # Test different cost ratios
    cost_ratios = [5.0, 10.0, 15.0, 20.0, 25.0]
    
    results_df = test_cost_ratios(cost_ratios)
    
    # Create comparison plots
    try:
        plot_cost_ratio_comparison(results_df)
    except Exception as e:
        logger.error(f"Error creating plots: {e}")
    
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT COMPLETE!")
    logger.info("="*70)
