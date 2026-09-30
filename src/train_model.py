import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from data_preprocessing import prepare_train_test_data, FEATURE_COLUMNS, FAILURE_MODES

def train_and_evaluate_models():
    print("Preparing training and test datasets...")
    data = prepare_train_test_data(filepath="predictive_maintenance.csv")
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    scaler = data['scaler']
    
    # Calculate scale_pos_weight for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    # Define models dictionary
    models = {
        'RandomForest_Balanced': RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight='balanced', random_state=42
        ),
        'XGBoost_Weighted': XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'
        ),
        'LogisticRegression_Balanced': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        )
    }
    
    # Apply SMOTE oversampling for an additional variant
    print("Applying SMOTE oversampling for SMOTE + XGBoost pipeline...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    models['XGBoost_SMOTE'] = XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.05,
        random_state=42, eval_metric='logloss'
    )
    
    results = {}
    fitted_models = {}
    
    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        if name == 'XGBoost_SMOTE':
            model.fit(X_train_smote, y_train_smote)
        else:
            model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4),
            'ROC_AUC': round(roc_auc, 4),
            'PR_AUC': round(pr_auc, 4),
            'Confusion_Matrix': cm
        }
        fitted_models[name] = model
        
        print(f"\nModel: {name}")
        print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        print(f"  F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
        print(f"  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # Select best model based on F1-Score / PR-AUC balance
    best_model_name = max(results.keys(), key=lambda k: results[k]['F1_Score'])
    best_model = fitted_models[best_model_name]
    print(f"\n[BEST MODEL SELECTED]: {best_model_name} (F1-Score: {results[best_model_name]['F1_Score']})")
    
    # Train failure mode specific classifiers for diagnostics
    print("\nTraining diagnostic classifiers for individual failure modes (TWF, HDF, PWF, OSF, RNF)...")
    df_processed = data['df_processed']
    mode_models = {}
    
    for mode in FAILURE_MODES:
        y_mode = df_processed.loc[X_train.index, mode]
        pos_m = (y_mode == 1).sum()
        neg_m = (y_mode == 0).sum()
        spw = (neg_m / pos_m) if pos_m > 0 else 1.0
        
        clf_mode = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            scale_pos_weight=spw, random_state=42, eval_metric='logloss'
        )
        clf_mode.fit(X_train, y_mode)
        mode_models[mode] = clf_mode

    # Extract feature importance from best model
    feature_importance = {}
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance = {col: round(float(imp), 4) for col, imp in zip(FEATURE_COLUMNS, importances)}
        # Sort by importance descending
        feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save artifacts
    joblib.dump(best_model, os.path.join("models", "best_model.joblib"))
    joblib.dump(scaler, os.path.join("models", "scaler.joblib"))
    joblib.dump(mode_models, os.path.join("models", "failure_mode_models.joblib"))
    
    # Save evaluation summary
    with open(os.path.join("models", "metrics_summary.json"), "w") as f:
        json.dump({
            'best_model_name': best_model_name,
            'all_results': results,
            'feature_importance': feature_importance,
            'feature_columns': FEATURE_COLUMNS
        }, f, indent=4)

    print("\nAll artifacts saved successfully to models/ directory!")
    return best_model, scaler, results

if __name__ == "__main__":
    train_and_evaluate_models()
