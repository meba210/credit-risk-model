import shap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def shap_explain(model: RandomForestClassifier, X: pd.DataFrame):
    """
    Generate SHAP values for model explainability and visualize them.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot for global feature importance
    shap.summary_plot(shap_values, X, plot_type="bar")

    # Example: local explanation for first instance
    shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X.iloc[0,:])
