"""
Model Training Script
Train Logistic Regression และ XGBoost models พร้อม 5-Fold Cross-Validation
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import xgboost as xgb

from data_prep import get_prepared_data
from imbalance_handlers import get_resampler
from logger_config import setup_logger
from config import (
    MODELS_DIR, PLOTS_DIR, RUN_NUMBER, CV_FOLDS, RANDOM_STATE,
    LR_MAX_ITER, LR_SOLVER,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_RANDOM_STATE,
    RESAMPLING_METHOD
)

logger = setup_logger("train_models")


def create_directories():
    """สร้างโฟลเดอร์สำหรับเก็บ models และ plots ตาม run number"""
    run_dir = f"run_{RUN_NUMBER}"
    models_dir = Path(MODELS_DIR) / run_dir
    plots_dir = Path(PLOTS_DIR) / run_dir
    
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directories for Run #{RUN_NUMBER}")
    logger.info(f"  Models: {models_dir}")
    logger.info(f"  Plots: {plots_dir}")
    
    return models_dir, plots_dir


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """คำนวณ metrics ทั้งหมด"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }


def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train Logistic Regression with class_weight='balanced'
    และทำ 5-Fold Cross-Validation
    """
    logger.info("="*70)
    logger.info("Training Logistic Regression")
    logger.info("="*70)
    
    # สร้าง model
    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=LR_MAX_ITER,
        solver=LR_SOLVER,
        random_state=RANDOM_STATE
    )
    logger.debug(f"Model params: {lr_model.get_params()}")
    
    # Cross-Validation
    logger.info(f"Performing {CV_FOLDS}-Fold Cross-Validation...")
    cv_scores = cross_validate(
        lr_model, X_train, y_train,
        cv=CV_FOLDS,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        return_train_score=False
    )
    
    logger.info("Cross-Validation Results:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        scores = cv_scores[f'test_{metric}']
        logger.info(f"  {metric.upper()}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # Train บน train set ทั้งหมด
    logger.info("Training on full training set...")
    lr_model.fit(X_train, y_train)
    
    # Evaluate บน validation set
    logger.info("Evaluating on validation set...")
    y_val_pred = lr_model.predict(X_val)
    y_val_proba = lr_model.predict_proba(X_val)[:, 1]
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
    
    logger.info("Validation Metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    # Evaluate บน test set
    logger.info("Evaluating on test set...")
    y_test_pred = lr_model.predict(X_test)
    y_test_proba = lr_model.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
    
    logger.info("Test Metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    return lr_model, cv_scores, val_metrics, test_metrics


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train XGBoost with scale_pos_weight
    และทำ 5-Fold Cross-Validation
    """
    logger.info("="*70)
    logger.info("Training XGBoost")
    logger.info("="*70)
    
    # คำนวณ scale_pos_weight
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")
    
    # สร้าง model
    xgb_model = xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        scale_pos_weight=scale_pos_weight,
        random_state=XGB_RANDOM_STATE,
        eval_metric='logloss'
    )
    logger.debug(f"Model params: {xgb_model.get_params()}")
    
    # Cross-Validation
    logger.info(f"Performing {CV_FOLDS}-Fold Cross-Validation...")
    cv_scores = cross_validate(
        xgb_model, X_train, y_train,
        cv=CV_FOLDS,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        return_train_score=False
    )
    
    logger.info("Cross-Validation Results:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        scores = cv_scores[f'test_{metric}']
        logger.info(f"  {metric.upper()}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # Train บน train set ทั้งหมด
    logger.info("Training on full training set...")
    xgb_model.fit(X_train, y_train)
    
    # Evaluate บน validation set
    logger.info("Evaluating on validation set...")
    y_val_pred = xgb_model.predict(X_val)
    y_val_proba = xgb_model.predict_proba(X_val)[:, 1]
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
    
    logger.info("Validation Metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    # Evaluate บน test set
    logger.info("Evaluating on test set...")
    y_test_pred = xgb_model.predict(X_test)
    y_test_proba = xgb_model.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
    
    logger.info("Test Metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    return xgb_model, cv_scores, val_metrics, test_metrics


def save_models(lr_model, xgb_model, preprocessor_lr, preprocessor_xgb, models_dir):
    """Save trained models and their respective preprocessors"""
    # Save Logistic Regression
    lr_path = models_dir / "logistic_regression.pkl"
    with open(lr_path, 'wb') as f:
        pickle.dump(lr_model, f)
    logger.info(f"Saved Logistic Regression to {lr_path}")
    
    # Save XGBoost
    xgb_path = models_dir / "xgboost.pkl"
    with open(xgb_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    logger.info(f"Saved XGBoost to {xgb_path}")
    
    # Save LR Preprocessor (with OneHot)
    prep_lr_path = models_dir / "preprocessor_lr.pkl"
    with open(prep_lr_path, 'wb') as f:
        pickle.dump(preprocessor_lr, f)
    logger.info(f"Saved LR Preprocessor to {prep_lr_path}")
    
    # Save XGBoost Preprocessor (with Label Encoding)
    prep_xgb_path = models_dir / "preprocessor_xgb.pkl"
    with open(prep_xgb_path, 'wb') as f:
        pickle.dump(preprocessor_xgb, f)
    logger.info(f"Saved XGBoost Preprocessor to {prep_xgb_path}")
    
    # Save run info
    info_path = models_dir / "run_info.txt"
    with open(info_path, 'w') as f:
        f.write(f"Run Number: {RUN_NUMBER}\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"\nConfiguration:\n")
        f.write(f"  LR max_iter: {LR_MAX_ITER}\n")
        f.write(f"  LR solver: {LR_SOLVER}\n")
        f.write(f"  XGB n_estimators: {XGB_N_ESTIMATORS}\n")
        f.write(f"  XGB max_depth: {XGB_MAX_DEPTH}\n")
        f.write(f"  XGB learning_rate: {XGB_LEARNING_RATE}\n")
        f.write(f"\nPreprocessing:\n")
        f.write(f"  LR: FixedBinnerForLR + OneHotEncoder\n")
        f.write(f"  XGBoost: FixedBinnerForXGBoost (Label Encoding)\n")
        f.write(f"\nImbalanced Data Handling:\n")
        f.write(f"  Resampling Method: {RESAMPLING_METHOD}\n")
    logger.info(f"Saved run info to {info_path}")


def print_comparison(lr_test_metrics, xgb_test_metrics):
    """พิมพ์ตารางเปรียบเทียบ models"""
    logger.info("="*70)
    logger.info("MODEL COMPARISON (Test Set)")
    logger.info("="*70)
    
    df = pd.DataFrame({
        'Logistic Regression': lr_test_metrics,
        'XGBoost': xgb_test_metrics
    })
    
    print("\n" + df.to_string())
    print("\n")


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("="*70)
    
    # สร้างโฟลเดอร์
    models_dir, plots_dir = create_directories()
    
    # โหลดและเตรียมข้อมูล
    logger.info("Loading and preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor_lr, preprocessor_xgb = get_prepared_data()
    
    # ===== Logistic Regression Pipeline =====
    logger.info("="*70)
    logger.info("LOGISTIC REGRESSION PIPELINE (with OneHot Encoding)")
    logger.info("="*70)
    
    # Fit preprocessor และ transform สำหรับ LR
    logger.info("Fitting LR preprocessor...")
    preprocessor_lr.fit(X_train, y_train)
    
    logger.info("Transforming datasets for LR...")
    X_train_lr = preprocessor_lr.transform(X_train)
    X_val_lr = preprocessor_lr.transform(X_val)
    X_test_lr = preprocessor_lr.transform(X_test)
    
    logger.info(f"LR Transformed shapes - Train: {X_train_lr.shape}, Val: {X_val_lr.shape}, Test: {X_test_lr.shape}")
    
    # Apply resampling if specified (เฉพาะ training set!)
    resampler = get_resampler(RESAMPLING_METHOD, random_state=RANDOM_STATE)
    X_train_lr_resampled, y_train_lr_resampled = resampler(X_train_lr, y_train)
    
    # Train Logistic Regression
    lr_model, lr_cv, lr_val_metrics, lr_test_metrics = train_logistic_regression(
        X_train_lr_resampled, y_train_lr_resampled,
        X_val_lr, y_val,
        X_test_lr, y_test
    )
    
    # ===== XGBoost Pipeline =====
    logger.info("="*70)
    logger.info("XGBOOST PIPELINE (with Label Encoding)")
    logger.info("="*70)
    
    # Fit preprocessor และ transform สำหรับ XGBoost
    logger.info("Fitting XGBoost preprocessor...")
    preprocessor_xgb.fit(X_train, y_train)
    
    logger.info("Transforming datasets for XGBoost...")
    X_train_xgb = preprocessor_xgb.transform(X_train)
    X_val_xgb = preprocessor_xgb.transform(X_val)
    X_test_xgb = preprocessor_xgb.transform(X_test)
    
    logger.info(f"XGBoost Transformed shapes - Train: {X_train_xgb.shape}, Val: {X_val_xgb.shape}, Test: {X_test_xgb.shape}")
    
    # Apply resampling if specified (เฉพาะ training set!)
    # ใช้ resampler เดียวกันกับ LR
    X_train_xgb_resampled, y_train_xgb_resampled = resampler(X_train_xgb, y_train)
    
    # Train XGBoost
    xgb_model, xgb_cv, xgb_val_metrics, xgb_test_metrics = train_xgboost(
        X_train_xgb_resampled, y_train_xgb_resampled,
        X_val_xgb, y_val,
        X_test_xgb, y_test
    )
    
    # Save models
    save_models(lr_model, xgb_model, preprocessor_lr, preprocessor_xgb, models_dir)
    
    # Print comparison
    print_comparison(lr_test_metrics, xgb_test_metrics)
    
    logger.info("="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Models saved to: {models_dir}")
    logger.info("="*70)

