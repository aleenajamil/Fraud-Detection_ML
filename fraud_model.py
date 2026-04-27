import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Running enhanced fraud detection model...")

# -----------------------------
# 1. Generate synthetic data
# -----------------------------
np.random.seed(42)
n = 500

amount = np.random.randint(1000, 50000, n)
avg_amount = np.random.randint(1000, 30000, n)
frequency = np.random.randint(1, 20, n)

# fraud label (rule-based for simulation)
fraud = ((amount > avg_amount * 2) | (amount > 40000)).astype(int)

df = pd.DataFrame({
    "amount": amount,
    "avg_amount": avg_amount,
    "frequency": frequency,
    "fraud": fraud
})

print("\nSample Data:")
print(df.head())

# -----------------------------
# 2. Feature Engineering (UPGRADE)
# -----------------------------
df["amount_to_avg_ratio"] = df["amount"] / (df["avg_amount"] + 1)
df["deviation"] = abs(df["amount"] - df["avg_amount"])
df["is_large_transaction"] = (df["amount"] > 30000).astype(int)
df["high_frequency_flag"] = (df["frequency"] > 10).astype(int)
df["risk_score_proxy"] = df["amount_to_avg_ratio"] * df["frequency"]

# -----------------------------
# 3. Features and Labels
# -----------------------------
X = df[[
    "amount",
    "avg_amount",
    "frequency",
    "amount_to_avg_ratio",
    "deviation",
    "is_large_transaction",
    "high_frequency_flag",
    "risk_score_proxy"
]]

y = df["fraud"]

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Logistic Regression Model
# -----------------------------
lr_model = LogisticRegression(class_weight="balanced")
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# -----------------------------
# 6. Random Forest Model
# -----------------------------
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# -----------------------------
# 7. Feature Importance
# -----------------------------
print("\n--- Feature Importance (Random Forest) ---")
for name, score in zip(X.columns, rf_model.feature_importances_):
    print(name, ":", round(score, 3))

# -----------------------------
# 8. Anomaly Detection (Unsupervised)
# -----------------------------
iso = IsolationForest(contamination=0.1, random_state=42)
df["anomaly"] = iso.fit_predict(X)

print("\n--- Anomaly Detection Sample ---")
print(df[["amount", "avg_amount", "anomaly"]].head())