
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# Dataset load karo
data_path = Path("data/processed/cicids2017_clean.csv")
data = pd.read_csv(data_path)

# Features (X) aur target label (y) separate karo
X = data.drop(columns=["Label"])
y = data["Label"]

print("Dataset shape:", data.shape)
print("Number of features:", X.shape[1])
print("Number of classes:", y.nunique())

# Training aur testing sets banao
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nModel training complete!")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# Model save karo
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "random_forest_model.joblib"

joblib.dump(
    {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "classes": model.classes_.tolist()
    },
    model_path
)

print("\nModel saved to:", model_path)

