
# 1. Import the libraries we need
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# 2. Load our dataset
data = pd.read_csv("data/demo.csv")


# 3. Separate input features and target label
X = data.drop(columns=["label"])
y = data["label"]


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# 5. Create the AI model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# 6. Train the model
model.fit(X_train, y_train)


# 7. Predict labels for unseen test records
predictions = model.predict(X_test)


# 8. Display evaluation results
print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))


# 9. Try a sample prediction
sample = X_test.iloc[[0]]
prediction = model.predict(sample)[0]

print("\nSample record:")
print(sample)

print("Predicted class:", prediction)
