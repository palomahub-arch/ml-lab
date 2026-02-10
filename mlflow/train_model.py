import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# conectar no MLflow server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud-detection")


# carregar dados processados pelo Spark
df = pd.read_parquet("spark/output")

# =========================
# FEATURE ENGINEERING
# =========================

df["high_amount"] = (df["amount"] > 300).astype(int)

df["user_avg"] = df.groupby("user_id")["amount"].transform("mean")

df["diff_from_avg"] = df["amount"] - df["user_avg"]

df["user_tx_count"] = df.groupby("user_id")["amount"].transform("count")


# =========================
# FEATURES + TARGET
# =========================

X = df[[
    "amount",
    "high_amount",
    "user_avg",
    "diff_from_avg",
    "user_tx_count"
]]

y = df["fraud"]


# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# TRAINING + MLFLOW LOGGING
# =========================

mlflow.set_experiment("fraud-detection")

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "fraud_model")

    print("✅ Modelo treinado com Feature Engineering!")
    print("Accuracy:", acc)