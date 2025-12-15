"""
Enhanced model training with feature selection, time series CV, ensemble stacking, and explainability.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from src.train_models import (
    load_processed_data, prepare_features_target,
    train_random_forest, train_xgboost, train_lightgbm,
    evaluate_model, sanitize_feature_names
)
from src.feature_selection import select_features_combined
from src.time_series_cv import TimeSeriesSplitByYear, create_time_series_splits
from src.ensemble_stacking import StackingEnsemble
from src.explainability import explain_with_shap, plot_shap_summary

import matplotlib.pyplot as plt


def train_with_time_series_cv(df: pd.DataFrame, feature_names: list,
                              use_ts_cv: bool = True) -> dict:
    """
    Train models using time series cross-validation
    
    Args:
        df: DataFrame with features and target
        feature_names: List of feature names
        use_ts_cv: Whether to use time series CV
        
    Returns:
        Dictionary with CV results
    """
    if not use_ts_cv or 'year' not in df.columns:
        return None
    
    print("\n" + "=" * 50)
    print("Time Series Cross-Validation")
    print("=" * 50)
    
    X = df[feature_names]
    y = df['yield']
    
    # Create time series splits
    splits = create_time_series_splits(df, year_col='year', n_splits=3)
    
    cv_results = {
        'random_forest': {'RMSE': [], 'R²': []},
        'xgboost': {'RMSE': [], 'R²': []},
        'lightgbm': {'RMSE': [], 'R²': []}
    }
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\nFold {fold + 1}/{len(splits)}")
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]
        
        # Train models on fold
        rf_model, rf_metrics, _ = train_random_forest(
            X_train_fold, y_train_fold, X_test_fold, y_test_fold, feature_names
        )
        cv_results['random_forest']['RMSE'].append(rf_metrics['RMSE'])
        cv_results['random_forest']['R²'].append(rf_metrics['R²'])
        
        xgb_model, xgb_metrics, _ = train_xgboost(
            X_train_fold, y_train_fold, X_test_fold, y_test_fold, feature_names
        )
        cv_results['xgboost']['RMSE'].append(xgb_metrics['RMSE'])
        cv_results['xgboost']['R²'].append(xgb_metrics['R²'])
        
        lgb_model, lgb_metrics, _ = train_lightgbm(
            X_train_fold, y_train_fold, X_test_fold, y_test_fold, feature_names
        )
        cv_results['lightgbm']['RMSE'].append(lgb_metrics['RMSE'])
        cv_results['lightgbm']['R²'].append(lgb_metrics['R²'])
    
    # Calculate average metrics
    print("\nTime Series CV Results (Average):")
    for model_name in cv_results:
        avg_rmse = np.mean(cv_results[model_name]['RMSE'])
        avg_r2 = np.mean(cv_results[model_name]['R²'])
        print(f"{model_name.upper()}:")
        print(f"  Avg RMSE: {avg_rmse:.4f}")
        print(f"  Avg R²: {avg_r2:.4f}")
    
    return cv_results


def train_ensemble_stacking(X_train, y_train, X_test, y_test, feature_names):
    """
    Train stacking ensemble
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
        
    Returns:
        Stacking ensemble and metrics
    """
    print("\n" + "=" * 50)
    print("Training Stacking Ensemble")
    print("=" * 50)
    
    # Prepare base models
    base_models = {
        'random_forest': RandomForestRegressor(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
        ),
        'xgboost': xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1
        ),
        'lightgbm': lgb.LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1
        )
    }
    
    # Create stacking ensemble
    stacking = StackingEnsemble(
        base_models=base_models,
        meta_model=Ridge(alpha=1.0),
        cv_folds=5
    )
    
    # Fit ensemble
    stacking.fit(X_train.values, y_train.values, X_test.values)
    
    # Evaluate
    test_metrics = stacking.evaluate(X_test.values, y_test.values)
    
    print(f"\nStacking Ensemble Performance:")
    print(f"  RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  R²: {test_metrics['R²']:.4f}")
    
    return stacking, test_metrics


def generate_explanations(models: dict, X_test: pd.DataFrame, 
                         feature_names: list, save_dir: str = 'model'):
    """
    Generate SHAP explanations for models
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        feature_names: List of feature names
        save_dir: Directory to save explanations
    """
    print("\n" + "=" * 50)
    print("Generating Model Explanations (SHAP)")
    print("=" * 50)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, model in models.items():
        try:
            print(f"\nExplaining {model_name}...")
            
            # Prepare data
            if isinstance(X_test, pd.DataFrame):
                X_sample = X_test[feature_names]
            else:
                X_sample = pd.DataFrame(X_test, columns=feature_names)
            
            # Generate SHAP explanation
            shap_result = explain_with_shap(model, X_sample, max_samples=100)
            
            # Save feature importance
            importance_path = f'{save_dir}/{model_name}_shap_importance.csv'
            shap_result['feature_importance'].to_csv(importance_path, index=False)
            print(f"  SHAP importance saved to {importance_path}")
            
            # Plot and save
            plot_path = f'{save_dir}/{model_name}_shap_summary.png'
            plot_shap_summary(shap_result, max_features=20, save_path=plot_path)
            
        except Exception as e:
            print(f"  Error explaining {model_name}: {e}")


def train_all_models_enhanced(data_path='data/processed/features.csv',
                              test_size=0.2,
                              random_state=42,
                              use_feature_selection=True,
                              feature_selection_method='importance',
                              n_features=25,
                              use_time_series_cv=True,
                              use_stacking=True,
                              generate_explanations_flag=True):
    """
    Enhanced training with all advanced features
    
    Args:
        data_path: Path to processed features
        test_size: Proportion of test set
        random_state: Random seed
        use_feature_selection: Whether to use feature selection
        feature_selection_method: Method for feature selection
        n_features: Number of features to select
        use_time_series_cv: Whether to use time series CV
        use_stacking: Whether to train stacking ensemble
        generate_explanations_flag: Whether to generate SHAP explanations
        
    Returns:
        Dictionary with all results
    """
    print("=" * 50)
    print("Crop Yield Prediction - Enhanced Model Training")
    print("=" * 50)
    
    # Load data
    df = load_processed_data(data_path)
    
    # Prepare features and target
    X, y, all_feature_names = prepare_features_target(df)
    
    # Feature selection
    selected_features = all_feature_names
    feature_importance_df = None
    
    if use_feature_selection:
        print("\n" + "=" * 50)
        print("Feature Selection")
        print("=" * 50)
        selected_features, feature_importance_df = select_features_combined(
            X, y, method=feature_selection_method, n_features=n_features
        )
        X = X[selected_features]
        print(f"Selected {len(selected_features)} features from {len(all_feature_names)}")
    
    # Time series cross-validation (if year column exists)
    ts_cv_results = None
    if use_time_series_cv and 'year' in df.columns:
        df_features = df[selected_features + ['yield', 'year']].copy()
        ts_cv_results = train_with_time_series_cv(df_features, selected_features, use_ts_cv=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train base models
    models = {}
    metrics = {}
    feature_importances = {}
    
    # Random Forest
    rf_model, rf_metrics, rf_importance = train_random_forest(
        X_train, y_train, X_test, y_test, selected_features
    )
    models['random_forest'] = rf_model
    metrics['random_forest'] = rf_metrics
    feature_importances['random_forest'] = rf_importance
    
    # XGBoost
    xgb_model, xgb_metrics, xgb_importance = train_xgboost(
        X_train, y_train, X_test, y_test, selected_features
    )
    models['xgboost'] = xgb_model
    metrics['xgboost'] = xgb_metrics
    feature_importances['xgboost'] = xgb_importance
    
    # LightGBM
    lgb_model, lgb_metrics, lgb_importance = train_lightgbm(
        X_train, y_train, X_test, y_test, selected_features
    )
    models['lightgbm'] = lgb_model
    metrics['lightgbm'] = lgb_metrics
    feature_importances['lightgbm'] = lgb_importance
    
    # Stacking ensemble
    stacking_model = None
    stacking_metrics = None
    if use_stacking:
        stacking_model, stacking_metrics = train_ensemble_stacking(
            X_train, y_train, X_test, y_test, selected_features
        )
        models['stacking'] = stacking_model
        metrics['stacking'] = stacking_metrics
    
    # Select best model (lowest RMSE)
    best_model_name = min(metrics.keys(), key=lambda k: metrics[k]['RMSE'])
    best_model = models[best_model_name]
    
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    for model_name, model_metrics in metrics.items():
        print(f"{model_name.upper()}:")
        print(f"  RMSE: {model_metrics['RMSE']:.4f}")
        print(f"  R²: {model_metrics['R²']:.4f}")
    
    print(f"\nBest Model: {best_model_name.upper()} (RMSE: {metrics[best_model_name]['RMSE']:.4f})")
    
    # Generate explanations
    if generate_explanations_flag:
        generate_explanations(models, X_test, selected_features)
    
    # Save models
    os.makedirs('model', exist_ok=True)
    joblib.dump(rf_model, 'model/random_forest_model.pkl')
    joblib.dump(xgb_model, 'model/xgboost_model.pkl')
    joblib.dump(lgb_model, 'model/lightgbm_model.pkl')
    if stacking_model:
        stacking_model.save('model/stacking_ensemble.pkl')
    joblib.dump(best_model, 'model/best_model.pkl')
    
    # Save feature names
    joblib.dump(selected_features, 'model/feature_names.pkl')
    if feature_importance_df is not None:
        feature_importance_df.to_csv('model/feature_selection_importance.csv', index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv('model/model_metrics.csv')
    
    # Save feature importances
    for model_name, importance_df in feature_importances.items():
        importance_df.to_csv(f'model/{model_name}_feature_importance.csv', index=False)
    
    print("\nAll models saved to model/ directory")
    
    return {
        'models': models,
        'metrics': metrics,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'feature_importances': feature_importances,
        'selected_features': selected_features,
        'ts_cv_results': ts_cv_results,
        'stacking_model': stacking_model,
        'stacking_metrics': stacking_metrics
    }


if __name__ == "__main__":
    # Train with all enhanced features
    results = train_all_models_enhanced(
        use_feature_selection=True,
        feature_selection_method='importance',
        n_features=25,
        use_time_series_cv=True,
        use_stacking=True,
        generate_explanations_flag=True
    )
    
    print("\n" + "=" * 50)
    print("Enhanced Training Complete!")
    print("=" * 50)



