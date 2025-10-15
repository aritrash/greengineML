# lighting_classifier.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# Load dataset
# -------------------------
print("[INFO] Loading dataset from datasets/esp32adc-ads1115adc.xlsx...")
df = pd.read_excel("datasets/esp32adc-ads1115adc.xlsx", header=None)
df.columns = ["esp32_adc", "time_of_day", "light_on"]

# -------------------------
# Convert ESP32 ADC → ADS1115 signed 16-bit equivalent
# -------------------------
df["ads1115_adc"] = ((df["esp32_adc"] / 4095.0) * 65535 - 32768).astype(int)

print("[INFO] Sample converted values:")
print(df.head())

# -------------------------
# Features & Labels
# -------------------------
X = df[["ads1115_adc", "time_of_day"]]
y = df["light_on"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Manual Random Forest
# -------------------------
n_estimators = 20  # change as needed
estimators = []
print(f"[INFO] Training Random Forest with {n_estimators} trees...")

for i in range(n_estimators):
    sample_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_sample, y_sample = X_train.iloc[sample_idx], y_train.iloc[sample_idx]
    
    tree = DecisionTreeClassifier(max_features="sqrt", random_state=i)
    tree.fit(X_sample, y_sample)
    estimators.append(tree)
    
    print(f"[TRAINING] Tree {i+1}/{n_estimators} trained.")

print("[INFO] All trees trained.")

# -------------------------
# Ensemble prediction functions
# -------------------------
def forest_predict(X):
    preds = np.array([estimator.predict(X) for estimator in estimators])
    final_preds = np.apply_along_axis(lambda row: np.bincount(row).argmax(), axis=0, arr=preds)
    return final_preds

def forest_predict_proba(X):
    probs = np.mean([estimator.predict_proba(X) for estimator in estimators], axis=0)
    return probs

# -------------------------
# Evaluate model
# -------------------------
y_pred = forest_predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n[RESULTS] Model Evaluation on Test Data:")
print(f"Accuracy: {acc:.2f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# Save model with metadata
# -------------------------
metadata = {
    "estimators": estimators,
    "feature_names": ["ads1115_adc", "time_of_day"],
    "class_labels": sorted(y.unique().tolist()),  # 0 = OFF, 1 = ON
    "training_info": {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "test_accuracy": acc,
        "n_estimators": n_estimators,
        "date_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": "Manual Random Forest (Decision Trees)"
    }
}

joblib.dump(metadata, "lighting_model.pkl")
print("[INFO] Model + metadata saved as lighting_model.pkl")

# -------------------------
# Visualizations
# -------------------------
importances = np.mean([tree.feature_importances_ for tree in estimators], axis=0)
plt.figure(figsize=(6,4))
plt.bar(["ADC Value", "Time of Day"], importances)
plt.title("Feature Importances")
plt.ylabel("Importance")
plt.show()

plt.figure(figsize=(6,4))
x_min, x_max = X["ads1115_adc"].min()-1000, X["ads1115_adc"].max()+1000
y_min, y_max = X["time_of_day"].min()-1, X["time_of_day"].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = forest_predict(pd.DataFrame({"ads1115_adc": xx.ravel(), "time_of_day": yy.ravel()}))
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
scatter = plt.scatter(X["ads1115_adc"], X["time_of_day"],
            c=y, cmap="coolwarm", edgecolor="k")
plt.xlabel("ADS1115 ADC Value")
plt.ylabel("Time of Day")
plt.title("Decision Boundary (Light ON/OFF)")
plt.legend(*scatter.legend_elements(), title="Light")
plt.show()

probs = forest_predict_proba(X_test)
confidences = np.max(probs, axis=1)
plt.figure(figsize=(6,4))
plt.hist(confidences, bins=10, edgecolor="black")
plt.xlabel("Prediction Confidence")
plt.ylabel("Frequency")
plt.title("Confidence of Model Predictions")
plt.show()
