

from pathlib import Path
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="XAI CyberGuard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ XAI CyberGuard")
st.subheader("Explainable AI-Based Cyber Threat Detection System")

st.write(
    "Classify network-flow records using a Random Forest model and interpret the model’s predictions through SHAP feature importance."
)

# Load saved model
model_path = Path("models/random_forest_model.joblib")
saved = joblib.load(model_path)

model = saved["model"]
feature_columns = saved["feature_columns"]

# Sidebar
st.sidebar.header("Project Information")
st.sidebar.write("Model: Random Forest")
st.sidebar.write(f"Features: {len(feature_columns)}")
st.sidebar.write(f"Classes: {len(saved['classes'])}")

# Dataset metrics
data_path = Path("data/processed/cicids2017_clean.csv")
data = pd.read_csv(data_path)

col1, col2, col3 = st.columns(3)

col1.metric("Cleaned Records", f"{len(data):,}")
col2.metric("Input Features", len(feature_columns))
col3.metric("Traffic Classes", data["Label"].nunique())

st.divider()

# SHAP feature importance
st.header("📊 Model Explainability")

importance_path = Path("results/shap_feature_importance.csv")
chart_path = Path("results/shap_feature_importance.png")

if importance_path.exists():
    importance = pd.read_csv(importance_path)
    st.write("Top 15 features by mean absolute SHAP value")
    st.bar_chart(
        importance.head(15).set_index("Feature")[
            "Mean_Absolute_SHAP"
        ]
    )
else:
    st.info("SHAP importance file not found yet.")

if chart_path.exists():
    st.image(
        str(chart_path),
        caption="SHAP Feature Importance"
    )

st.divider()

# Sample prediction
st.header("🔎 Test a Dataset Record")

st.write(
    "This demo makes predictions on a single record from a cleaned dataset. It does not perform live network monitoring."
)

if st.button("Run sample prediction"):
    sample = data.sample(n=1, random_state=None)

    actual_label = sample["Label"].iloc[0]
    input_row = sample[feature_columns]

    prediction = model.predict(input_row)[0]
    
    
    # SHAP explanation for this prediction
    st.subheader("🔍 Why did the model make this prediction?")

    with st.spinner("Generating SHAP explanation..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_row)

        if isinstance(shap_values, list):
            values = np.stack(shap_values, axis=-1)
        else:
            values = np.asarray(shap_values)

        class_index = list(model.classes_).index(prediction)

        if values.ndim == 3:
            contributions = values[0, :, class_index]
        elif values.ndim == 2:
            contributions = values[0]
        else:
            st.error(f"Unexpected SHAP output shape: {values.shape}")
            st.stop()

        explanation = pd.DataFrame({
            "Feature": feature_columns,
            "SHAP contribution": contributions
        })

        explanation["Absolute contribution"] = (
            explanation["SHAP contribution"].abs()
        )

        top_features = explanation.nlargest(
            10, "Absolute contribution"
        ).sort_values("SHAP contribution")

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(
            top_features["Feature"],
            top_features["SHAP contribution"]
        )
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("SHAP contribution")
        ax.set_title(f"Explanation for: {prediction}")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Positive values support the selected class output; "
            "negative values oppose it. These are model explanations, "
            "not proof that an attack occurred."
        )


    st.metric("Predicted Class", str(prediction))
    st.write("Actual dataset label:", actual_label)

    if prediction == actual_label:
        st.success("Prediction matches the dataset label.")
    else:
        st.warning("Prediction differs from the dataset label.")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_row)[0]

        probability_df = pd.DataFrame({
            "Class": model.classes_,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        st.subheader("Top 5 Class Probabilities")
        st.dataframe(
            probability_df.head(5),
            use_container_width=True
        )

st.caption(
    "Major Project prototype (Project ID- 26P_3CSE122) | CICIDS2017 Dataset | Random Forest + SHAP"
)
