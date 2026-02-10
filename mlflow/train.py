import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud-detection")

df = pd.read_parquet("spark/output")

print(f"📊 Dataset: {df.shape}")
print(f"% Fraudes: {df['fraud'].mean()*100:.2f}%\n")

# =========================
# FEATURE ENGINEERING (mesmo código)
# =========================

df["high_amount"] = (df["amount"] > 300).astype(int)
df["user_avg"] = df.groupby("user_id")["amount"].transform("mean")
df["diff_from_avg"] = df["amount"] - df["user_avg"]
df["user_tx_count"] = df.groupby("user_id")["amount"].transform("count")

df["amount_log"] = np.log1p(df["amount"])
df["user_std"] = df.groupby("user_id")["amount"].transform("std").fillna(0)
df["user_min"] = df.groupby("user_id")["amount"].transform("min")
df["user_max"] = df.groupby("user_id")["amount"].transform("max")
df["user_median"] = df.groupby("user_id")["amount"].transform("median")

df["z_score"] = (df["amount"] - df["user_avg"]) / (df["user_std"] + 1e-5)
df["is_outlier"] = (df["z_score"].abs() > 3).astype(int)

df["pct_of_user_max"] = df["amount"] / (df["user_max"] + 1e-5)
df["diff_from_median"] = abs(df["amount"] - df["user_median"])
df["above_avg"] = (df["amount"] > df["user_avg"]).astype(int)

df["amount_bin"] = pd.cut(df["amount"], bins=[0, 50, 150, 300, 500, np.inf], 
                          labels=[0, 1, 2, 3, 4]).astype(int)

df["amount_x_count"] = df["amount"] * df["user_tx_count"]
df["diff_x_count"] = df["diff_from_avg"] * df["user_tx_count"]

feature_cols = [
    "amount", "high_amount", "user_avg", "diff_from_avg", "user_tx_count",
    "amount_log", "user_std", "user_min", "user_max", "user_median",
    "z_score", "is_outlier", "pct_of_user_max", "diff_from_median",
    "above_avg", "amount_bin", "amount_x_count", "diff_x_count"
]

X = df[feature_cols]
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# TRAINING + THRESHOLD OPTIMIZATION
# =========================

with mlflow.start_run(run_name="fraud-detection-optimized"):

    # Modelo otimizado
    model = RandomForestClassifier(
        n_estimators=500,           # 🔥 Mais árvores
        max_depth=15,               # 🔥 Mais profundo
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',        # 🔥 Evita overfitting
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Probabilidades
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Encontrar melhor threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    # F1-Score para cada threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    
    print(f"\n🎯 Melhor threshold: {best_threshold:.3f} (padrão: 0.500)")
    
    # Predições com threshold otimizado
    y_pred_optimized = (y_proba >= best_threshold).astype(int)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred_optimized)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Log
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("max_depth", 15)
    mlflow.log_param("threshold", best_threshold)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", roc_auc)
    
    mlflow.sklearn.log_model(model, "fraud_model")

    print(f"\n✅ Modelo Otimizado!")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    print("\n📊 Classification Report (threshold otimizado):")
    print(classification_report(y_test, y_pred_optimized, 
                                target_names=['Normal', 'Fraud']))
    
    print("\n🔍 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_optimized)
    print(cm)
    
    # Calcular métricas importantes
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📈 Métricas Detalhadas:")
    print(f"True Positives (fraudes detectadas): {tp}")
    print(f"False Negatives (fraudes perdidas): {fn} ⚠️")
    print(f"False Positives (alarmes falsos): {fp}")
    print(f"True Negatives (normais corretas): {tn}")
    print(f"\nRecall de Fraude: {tp/(tp+fn)*100:.1f}% das fraudes detectadas")
    print(f"Precision de Fraude: {tp/(tp+fp)*100:.1f}% dos alertas são corretos")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 Top 5 Features:")
    print(feature_importance.head(5))
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.plot(recalls[best_threshold_idx], precisions[best_threshold_idx], 
             'ro', markersize=10, label=f'Best threshold: {best_threshold:.3f}')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('precision_recall_curve.png', dpi=150, bbox_inches='tight')
    mlflow.log_artifact('precision_recall_curve.png')
    print("\n📊 Curva Precision-Recall salva!")

p