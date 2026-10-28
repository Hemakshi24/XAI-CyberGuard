
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Saved model load karo
saved = joblib.load("models/random_forest_model.joblib")
model = saved["model"]
feature_columns = saved["feature_columns"]

# Clean dataset load karo
data = pd.read_csv("data/processed/cicids2017_clean.csv")

X = data[feature_columns]

# Explanation ke liye 100 records ka sample
X_sample = X.sample(
    n=min(100, len(X)),
    random_state=42
)

print("Generating SHAP explanations...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Different SHAP versions ke output ko handle karo
if isinstance(shap_values, list):
    values = np.stack(shap_values, axis=-1)
else:
    values = np.asarray(shap_values)

# Har feature ka average absolute SHAP importance
if values.ndim == 3:
    importance = np.abs(values).mean(axis=(0, 2))
elif values.ndim == 2:
    importance = np.abs(values).mean(axis=0)
else:
    raise ValueError(
        f"Unexpected SHAP output shape: {values.shape}"
    )

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Mean_Absolute_SHAP": importance
}).sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

# Results save karo
output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

importance_df.to_csv(
    output_dir / "shap_feature_importance.csv",
    index=False
)

# Top 15 features ka graph
top_features = importance_df.head(15).sort_values(
    "Mean_Absolute_SHAP"
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["Mean_Absolute_SHAP"]
)
plt.xlabel("Mean absolute SHAP value")
plt.title("Top 15 Features - SHAP Importance")
plt.tight_layout()

plt.savefig(
    output_dir / "shap_feature_importance.png",
    dpi=300
)
plt.show()

print("\nSHAP explanation complete!")
print("\nTop 15 important features:")
print(importance_df.head(15).to_string(index=False))
print("\nSaved results in:", output_dir)
