
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

# Ek test record select karo
row = data.sample(n=1, random_state=42)
actual_label = row["Label"].iloc[0]
X_one = row[feature_columns]

# Prediction
prediction = model.predict(X_one)[0]

print("Actual label:", actual_label)
print("Predicted label:", prediction)

# SHAP explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_one)

# Output format handle karo
if isinstance(shap_values, list):
    values = np.stack(shap_values, axis=-1)
else:
    values = np.asarray(shap_values)

# Selected prediction class ke SHAP values
class_index = list(model.classes_).index(prediction)

if values.ndim == 3:
    contributions = values[0, :, class_index]
elif values.ndim == 2:
    contributions = values[0]
else:
    raise ValueError(f"Unexpected SHAP shape: {values.shape}")

# Feature contributions
explanation = pd.DataFrame({
    "Feature": feature_columns,
    "Value": X_one.iloc[0].values,
    "SHAP_Value": contributions
})

explanation["Absolute_SHAP"] = explanation["SHAP_Value"].abs()
explanation = explanation.sort_values(
    "Absolute_SHAP",
    ascending=False
)

print("\nTop 10 features for this prediction:")
print(
    explanation[
        ["Feature", "Value", "SHAP_Value"]
    ].head(10).to_string(index=False)
)

# Graph save karo
output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

top = explanation.head(10).sort_values("SHAP_Value")

plt.figure(figsize=(10, 6))
plt.barh(top["Feature"], top["SHAP_Value"])
plt.axvline(0, linewidth=1)
plt.xlabel("SHAP contribution")
plt.title(f"Feature Contributions for Prediction: {prediction}")
plt.tight_layout()

output_file = output_dir / "individual_prediction_shap.png"
plt.savefig(output_file, dpi=300)
plt.show()

print("\nExplanation graph saved to:", output_file)
