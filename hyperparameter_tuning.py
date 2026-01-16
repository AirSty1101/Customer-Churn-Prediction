"""
Hyperparameter Tuning for XGBoost (Run #2)
Goal: Fine-tune XGBoost parameters to potentially improve ROC-AUC and other metrics
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    make_scorer
)
import warnings
warnings.filterwarnings('ignore')

# Import from existing modules
from data_prep import get_prepared_data
from imbalance_handlers import get_resampler
from config import RANDOM_STATE, RESAMPLING_METHOD

# Load the trained model
print("Loading trained XGBoost model...")
with open('models/run_2/xgboost.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Load and prepare data using the same pipeline as train_models.py
print("Loading and preparing data using the same pipeline as train_models.py...")
X_train, X_val, X_test, y_train, y_val, y_test, preprocessor_lr, preprocessor_xgb = get_prepared_data()

# Fit preprocessor and transform for XGBoost (same as train_models.py)
print("Fitting XGBoost preprocessor...")
preprocessor_xgb.fit(X_train, y_train)

print("Transforming datasets for XGBoost...")
X_train_xgb = preprocessor_xgb.transform(X_train)
X_val_xgb = preprocessor_xgb.transform(X_val)
X_test_xgb = preprocessor_xgb.transform(X_test)

print(f"Transformed shapes - Train: {X_train_xgb.shape}, Val: {X_val_xgb.shape}, Test: {X_test_xgb.shape}")

# Apply resampling if specified (same as train_models.py)
resampler = get_resampler(RESAMPLING_METHOD, random_state=RANDOM_STATE)
X_train_processed, y_train_processed = resampler(X_train_xgb, y_train)

print(f"After resampling - Train: {X_train_processed.shape}")
print(f"Class distribution - 0: {(y_train_processed == 0).sum()}, 1: {(y_train_processed == 1).sum()}")

# Define parameter grid for GridSearch
print("\n" + "="*80)
print("HYPERPARAMETER TUNING - XGBoost (Run #2)")
print("="*80)

# Option 1: GridSearchCV (exhaustive search - more accurate but slower)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Option 2: RandomizedSearchCV (faster - good for initial exploration)
param_distributions = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
    'min_child_weight': [1, 2, 3, 4, 5],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.5],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.1, 0.5, 1.0, 2.0]
}

# Choose search method
USE_RANDOMIZED_SEARCH = True  # Set to False for GridSearchCV

if USE_RANDOMIZED_SEARCH:
    print("\nUsing RandomizedSearchCV (faster)...")
    print(f"Testing {len(param_distributions)} parameters")
    print("This will test 50 random combinations\n")
    
    # Custom scorer - prioritize Recall but also consider F1
    def custom_scorer(y_true, y_pred):
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        # Weighted score: 60% Recall + 40% F1
        return 0.6 * recall + 0.4 * f1
    
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=50,  # Number of random combinations to try
        scoring=make_scorer(custom_scorer),
        cv=3,  # 3-fold CV for speed
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
else:
    print("\nUsing GridSearchCV (exhaustive but slower)...")
    print(f"Testing all combinations of {len(param_grid)} parameters")
    print("This may take a while...\n")
    
    search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='recall',  # Optimize for Recall
        cv=3,
        verbose=2,
        n_jobs=-1
    )

# Fit the search
print("Starting hyperparameter search...")
print("This may take 10-30 minutes depending on your machine...\n")

search.fit(X_train_processed, y_train_processed)

# Get best parameters
print("\n" + "="*80)
print("BEST PARAMETERS FOUND:")
print("="*80)
for param, value in search.best_params_.items():
    print(f"{param:20s}: {value}")

# Train model with best parameters
print("\n" + "="*80)
print("EVALUATING BEST MODEL")
print("="*80)

best_model = search.best_estimator_

# Predictions
y_train_pred = best_model.predict(X_train_processed)
y_val_pred = best_model.predict(X_val_xgb)
y_test_pred = best_model.predict(X_test_xgb)

y_train_proba = best_model.predict_proba(X_train_processed)[:, 1]
y_val_proba = best_model.predict_proba(X_val_xgb)[:, 1]
y_test_proba = best_model.predict_proba(X_test_xgb)[:, 1]

# Calculate metrics
def calculate_metrics(y_true, y_pred, y_proba, dataset_name):
    print(f"\n{dataset_name} Set:")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")

calculate_metrics(y_train_processed, y_train_pred, y_train_proba, "Training")
calculate_metrics(y_val, y_val_pred, y_val_proba, "Validation")
calculate_metrics(y_test, y_test_pred, y_test_proba, "Test")

# Compare with baseline (Run #2)
print("\n" + "="*80)
print("COMPARISON WITH BASELINE (Run #2)")
print("="*80)

baseline_metrics = {
    'Accuracy': 0.7880,
    'Precision': 0.4862,
    'Recall': 0.6895,
    'F1': 0.5703,
    'ROC-AUC': 0.8379
}

tuned_metrics = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'Precision': precision_score(y_test, y_test_pred),
    'Recall': recall_score(y_test, y_test_pred),
    'F1': f1_score(y_test, y_test_pred),
    'ROC-AUC': roc_auc_score(y_test, y_test_proba)
}

print(f"\n{'Metric':<12} {'Baseline':<12} {'Tuned':<12} {'Change':<12}")
print("-" * 50)
for metric in baseline_metrics.keys():
    baseline = baseline_metrics[metric]
    tuned = tuned_metrics[metric]
    change = tuned - baseline
    change_pct = (change / baseline) * 100
    print(f"{metric:<12} {baseline:<12.4f} {tuned:<12.4f} {change:+.4f} ({change_pct:+.2f}%)")

# Save best model
print("\n" + "="*80)
print("SAVING BEST MODEL")
print("="*80)

import os
os.makedirs('models/run_2_tuned', exist_ok=True)

with open('models/run_2_tuned/xgb_model_tuned.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save best parameters
with open('models/run_2_tuned/best_params.txt', 'w') as f:
    f.write("Best Hyperparameters:\n")
    f.write("=" * 50 + "\n")
    for param, value in search.best_params_.items():
        f.write(f"{param}: {value}\n")
    
    f.write("\n\nTest Set Metrics:\n")
    f.write("=" * 50 + "\n")
    for metric, value in tuned_metrics.items():
        f.write(f"{metric}: {value:.4f}\n")
    
    f.write("\n\nComparison with Baseline:\n")
    f.write("=" * 50 + "\n")
    for metric in baseline_metrics.keys():
        baseline = baseline_metrics[metric]
        tuned = tuned_metrics[metric]
        change = tuned - baseline
        change_pct = (change / baseline) * 100
        f.write(f"{metric}: {baseline:.4f} -> {tuned:.4f} ({change_pct:+.2f}%)\n")

print("\n✅ Best model saved to: models/run_2_tuned/xgb_model_tuned.pkl")
print("✅ Best parameters saved to: models/run_2_tuned/best_params.txt")

# Save detailed results
results_df = pd.DataFrame(search.cv_results_)
results_df.to_csv('models/run_2_tuned/cv_results.csv', index=False)
print("✅ CV results saved to: models/run_2_tuned/cv_results.csv")

print("\n" + "="*80)
print("HYPERPARAMETER TUNING COMPLETED!")
print("="*80)

# Recommendations
print("\n📊 RECOMMENDATIONS:")
if tuned_metrics['ROC-AUC'] > baseline_metrics['ROC-AUC']:
    print("✅ Tuned model shows improvement in ROC-AUC")
    print("   → Consider using the tuned model for production")
else:
    print("⚠️  Tuned model did not improve ROC-AUC significantly")
    print("   → Baseline model may be sufficient")

if tuned_metrics['Recall'] > baseline_metrics['Recall']:
    print("✅ Tuned model shows improvement in Recall")
else:
    print("⚠️  Tuned model did not improve Recall")
    print("   → Consider threshold tuning instead")

print("\n💡 Next Steps:")
print("1. Review the best parameters above")
print("2. Check if there are signs of overfitting (Training vs Test metrics)")
print("3. If satisfied, update RESULTS.md with the tuned model results")
print("4. Consider combining with threshold tuning for optimal performance")
