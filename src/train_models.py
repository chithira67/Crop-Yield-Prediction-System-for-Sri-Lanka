import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def load_processed_data(file_path='data/processed/features.csv'):
    """
    Load processed features from CSV
    
    Args:
        file_path: Path to processed features CSV
        
    Returns:
        DataFrame with features and target
    """
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} samples from {file_path}")
    return df


def prepare_features_target(df):
    """
    Separate features and target variable
    
    Args:
        df: DataFrame with features and target
        
    Returns:
        X (features), y (target)
    """
    # Exclude target and original categorical columns
    exclude_cols = ['yield', 'crop', 'district']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['yield']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    return X, y, feature_cols


def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate model performance using RMSE and R²
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        model_name: Name of the model
        
    Returns:
        Dictionary with metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'R²': r2
    }
    
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return metrics


def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    """
    Train Random Forest model with hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
        
    Returns:
        Trained model and metrics
    """
    print("\n" + "=" * 50)
    print("Training Random Forest Model")
    print("=" * 50)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Base model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Grid search with cross-validation
    print("Performing hyperparameter tuning...")
    grid_search = RandomizedSearchCV(
        rf, param_grid, 
        n_iter=20,  # Reduced for faster training
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_rf = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")
    
    # Predictions
    y_train_pred = best_rf.predict(X_train)
    y_test_pred = best_rf.predict(X_test)
    
    # Evaluation
    train_metrics = evaluate_model(y_train, y_train_pred, "Random Forest (Train)")
    test_metrics = evaluate_model(y_test, y_test_pred, "Random Forest (Test)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_rf, test_metrics, feature_importance


def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    """
    Train XGBoost model with hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
        
    Returns:
        Trained model and metrics
    """
    print("\n" + "=" * 50)
    print("Training XGBoost Model")
    print("=" * 50)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Base model
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    # Randomized search
    print("Performing hyperparameter tuning...")
    random_search = RandomizedSearchCV(
        xgb_model, param_grid,
        n_iter=20,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    # Best model
    best_xgb = random_search.best_estimator_
    print(f"\nBest parameters: {random_search.best_params_}")
    
    # Predictions
    y_train_pred = best_xgb.predict(X_train)
    y_test_pred = best_xgb.predict(X_test)
    
    # Evaluation
    train_metrics = evaluate_model(y_train, y_train_pred, "XGBoost (Train)")
    test_metrics = evaluate_model(y_test, y_test_pred, "XGBoost (Test)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_xgb, test_metrics, feature_importance


def train_lightgbm(X_train, y_train, X_test, y_test, feature_names):
    """
    Train LightGBM model with hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
        
    Returns:
        Trained model and metrics
    """
    print("\n" + "=" * 50)
    print("Training LightGBM Model")
    print("=" * 50)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'num_leaves': [31, 50, 100]
    }
    
    # Base model
    lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    
    # Randomized search
    print("Performing hyperparameter tuning...")
    random_search = RandomizedSearchCV(
        lgb_model, param_grid,
        n_iter=20,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    # Best model
    best_lgb = random_search.best_estimator_
    print(f"\nBest parameters: {random_search.best_params_}")
    
    # Predictions
    y_train_pred = best_lgb.predict(X_train)
    y_test_pred = best_lgb.predict(X_test)
    
    # Evaluation
    train_metrics = evaluate_model(y_train, y_train_pred, "LightGBM (Train)")
    test_metrics = evaluate_model(y_test, y_test_pred, "LightGBM (Test)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_lgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_lgb, test_metrics, feature_importance


def train_all_models(data_path='data/processed/features.csv', 
                     test_size=0.2, 
                     random_state=42):
    """
    Train all models and select the best one
    
    Args:
        data_path: Path to processed features
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Dictionary with all models, metrics, and best model
    """
    print("=" * 50)
    print("Crop Yield Prediction - Model Training")
    print("=" * 50)
    
    # Load data
    df = load_processed_data(data_path)
    
    # Prepare features and target
    X, y, feature_names = prepare_features_target(df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train all models
    models = {}
    metrics = {}
    feature_importances = {}
    
    # Random Forest
    rf_model, rf_metrics, rf_importance = train_random_forest(
        X_train, y_train, X_test, y_test, feature_names
    )
    models['random_forest'] = rf_model
    metrics['random_forest'] = rf_metrics
    feature_importances['random_forest'] = rf_importance
    
    # XGBoost
    xgb_model, xgb_metrics, xgb_importance = train_xgboost(
        X_train, y_train, X_test, y_test, feature_names
    )
    models['xgboost'] = xgb_model
    metrics['xgboost'] = xgb_metrics
    feature_importances['xgboost'] = xgb_importance
    
    # LightGBM
    lgb_model, lgb_metrics, lgb_importance = train_lightgbm(
        X_train, y_train, X_test, y_test, feature_names
    )
    models['lightgbm'] = lgb_model
    metrics['lightgbm'] = lgb_metrics
    feature_importances['lightgbm'] = lgb_importance
    
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
    
    # Save models
    os.makedirs('model', exist_ok=True)
    joblib.dump(rf_model, 'model/random_forest_model.pkl')
    joblib.dump(xgb_model, 'model/xgboost_model.pkl')
    joblib.dump(lgb_model, 'model/lightgbm_model.pkl')
    joblib.dump(best_model, 'model/best_model.pkl')
    
    # Save feature names
    joblib.dump(feature_names, 'model/feature_names.pkl')
    
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
        'feature_names': feature_names
    }


if __name__ == "__main__":
    # Train all models
    results = train_all_models()
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)

