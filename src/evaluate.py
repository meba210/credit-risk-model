import shap
import pandas as pd
import matplotlib.pyplot as plt

def generate_shap_plots(model, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Global importance
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("shap_summary.png")
    plt.close()

    # Local explanation for first instance
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][0],
        X.iloc[0],
        matplotlib=True
    )
    plt.savefig("shap_local.png")
    plt.close()
