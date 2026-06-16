import os
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "ocr_tabular_dataset.csv"
)

MODEL_SAVE_PATH = os.path.join(
    BASE_DIR,
    "catboost_ocr.cbm"
)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(DATASET_PATH)

print(df.head())

# =========================
# FEATURES
# =========================
feature_columns = [
    "text_length",

    "digit_count",
    "alpha_count",
    "space_count",

    "digit_ratio",
    "alpha_ratio",
    "space_ratio",

    "unique_chars",

    "repeat_ratio",

    "avg_confidence",
    "max_confidence",
    "min_confidence",
    "std_confidence"
]

# =========================
# TARGET
# =========================
target_column = "cer"

X = df[feature_columns]
y = df[target_column]

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL
# =========================
model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    verbose=50
)

# =========================
# TRAIN
# =========================
model.fit(
    X_train,
    y_train
)

# =========================
# PREDICT
# =========================
preds = model.predict(X_test)

# =========================
# METRICS
# =========================
mae = mean_absolute_error(y_test, preds)

rmse = mean_squared_error(
    y_test,
    preds
) ** 0.5

print()
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# =========================
# FEATURE IMPORTANCE
# =========================
importance = model.get_feature_importance()

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": importance
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print()
print(importance_df)

# =========================
# VISUALIZATION
# =========================
plt.figure(figsize=(10, 6))

plt.bar(
    importance_df["feature"],
    importance_df["importance"]
)

plt.xticks(rotation=45)

plt.title("CatBoost Feature Importance")

plt.tight_layout()
plt.show()

# =========================
# SAVE MODEL
# =========================
model.save_model(MODEL_SAVE_PATH)

print()
print(f"MODEL SAVED: {MODEL_SAVE_PATH}")