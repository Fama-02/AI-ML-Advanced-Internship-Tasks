import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv("telco.csv")
print("Dataset shape:", df.shape)

# ── 2. Preprocessing ──────────────────────────────────────────
# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges (it has spaces)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Encode all object columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

print("After preprocessing shape:", df.shape)

# ── 3. Split ──────────────────────────────────────────────────
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Logistic Regression Pipeline ───────────────────────────
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

print("\n── Logistic Regression ──")
print("Accuracy:", round(accuracy_score(y_test, lr_pred), 4))
print("F1 Score:", round(f1_score(y_test, lr_pred), 4))

# ── 5. Random Forest Pipeline ─────────────────────────────────
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

print("\n── Random Forest ──")
print("Accuracy:", round(accuracy_score(y_test, rf_pred), 4))
print("F1 Score:", round(f1_score(y_test, rf_pred), 4))

# ── 6. GridSearchCV on Random Forest ─────────────────────────
params = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [5, 10]
}

grid = GridSearchCV(rf_pipeline, params, cv=3, scoring="f1", verbose=1)
grid.fit(X_train, y_train)

print("\n── Best Params ──")
print(grid.best_params_)

best_pred = grid.best_estimator_.predict(X_test)
print("\n── Best Model Results ──")
print("Accuracy:", round(accuracy_score(y_test, best_pred), 4))
print("F1 Score:", round(f1_score(y_test, best_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, best_pred))

# ── 7. Confusion Matrix Plot ──────────────────────────────────
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\n✅ Confusion matrix saved")

# ── 8. Save Model ─────────────────────────────────────────────
joblib.dump(grid.best_estimator_, "model.pkl")
print("✅ Model saved as model.pkl")