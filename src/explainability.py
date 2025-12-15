"""
Model explainability using SHAP and LIME.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. Install with: pip install lime")


def explain_with_shap(model: Any, X: pd.DataFrame, X_sample: Optional[pd.DataFrame] = None,
                      max_samples: int = 100) -> Dict[str, Any]:
    """
    Generate SHAP explanations for model predictions
    
    Args:
        model: Trained model
        X: Feature dataframe
        X_sample: Sample to explain (if None, uses X)
        max_samples: Maximum samples for SHAP (for performance)
        
    Returns:
        Dictionary with SHAP values and plots
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    # Sample data if too large
    if len(X) > max_samples:
        X_sample_shap = X.sample(n=max_samples, random_state=42)
    else:
        X_sample_shap = X
    
    # Create SHAP explainer based on model type
    model_type = type(model).__name__
    
    if 'XGBoost' in model_type or 'XGB' in model_type:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_shap)
    elif 'LightGBM' in model_type or 'LGBM' in model_type:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_shap)
    elif 'RandomForest' in model_type or 'Forest' in model_type:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_shap)
    else:
        # Use KernelExplainer for other models (slower)
        explainer = shap.KernelExplainer(model.predict, X_sample_shap)
        shap_values = explainer.shap_values(X_sample_shap.iloc[:50])  # Limit for performance
    
    # Calculate feature importance
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    
    feature_importance = pd.DataFrame({
        'feature': X_sample_shap.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'feature_importance': feature_importance,
        'X_sample': X_sample_shap
    }


def plot_shap_summary(shap_result: Dict[str, Any], max_features: int = 20, 
                     save_path: Optional[str] = None):
    """
    Plot SHAP summary
    
    Args:
        shap_result: Result from explain_with_shap
        max_features: Maximum features to show
        save_path: Path to save plot (optional)
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available for plotting")
        return
    
    explainer = shap_result['explainer']
    shap_values = shap_result['shap_values']
    X_sample = shap_result['X_sample']
    
    # Plot summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=max_features)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP plot saved to {save_path}")
    else:
        plt.show()


def explain_with_lime(model: Any, X: pd.DataFrame, instance_idx: int = 0,
                     num_features: int = 10) -> Dict[str, Any]:
    """
    Generate LIME explanation for a single instance
    
    Args:
        model: Trained model
        X: Feature dataframe
        instance_idx: Index of instance to explain
        num_features: Number of features to show in explanation
        
    Returns:
        Dictionary with LIME explanation
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME is not installed. Install with: pip install lime")
    
    # Get instance to explain
    instance = X.iloc[instance_idx].values.reshape(1, -1)
    
    # Create LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X.values,
        feature_names=X.columns.tolist(),
        mode='regression',
        discretize_continuous=True
    )
    
    # Generate explanation
    explanation = explainer.explain_instance(
        X.iloc[instance_idx].values,
        model.predict,
        num_features=num_features
    )
    
    # Extract feature importance
    feature_importance = pd.DataFrame(
        explanation.as_list(),
        columns=['feature', 'importance']
    ).sort_values('importance', key=abs, ascending=False)
    
    return {
        'explainer': explainer,
        'explanation': explanation,
        'feature_importance': feature_importance,
        'instance': X.iloc[instance_idx],
        'prediction': model.predict(instance)[0]
    }


def plot_lime_explanation(lime_result: Dict[str, Any], save_path: Optional[str] = None):
    """
    Plot LIME explanation
    
    Args:
        lime_result: Result from explain_with_lime
        save_path: Path to save plot (optional)
    """
    if not LIME_AVAILABLE:
        print("LIME not available for plotting")
        return
    
    explanation = lime_result['explanation']
    
    # Create figure
    fig = explanation.as_pyplot_figure()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"LIME plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def compare_model_explanations(models: Dict[str, Any], X: pd.DataFrame,
                              instance_idx: int = 0) -> pd.DataFrame:
    """
    Compare explanations across multiple models
    
    Args:
        models: Dictionary of models {name: model}
        X: Feature dataframe
        instance_idx: Index of instance to explain
        
    Returns:
        DataFrame comparing feature importance across models
    """
    comparisons = []
    
    for model_name, model in models.items():
        try:
            if SHAP_AVAILABLE:
                shap_result = explain_with_shap(model, X, max_samples=50)
                # Get SHAP values for this instance
                instance_shap = shap_result['shap_values'][instance_idx]
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': np.abs(instance_shap)
                }).sort_values('importance', ascending=False)
                
                comparisons.append({
                    'model': model_name,
                    'method': 'SHAP',
                    'feature_importance': feature_importance
                })
        except Exception as e:
            print(f"Error explaining {model_name} with SHAP: {e}")
    
    return comparisons



