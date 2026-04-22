
import pandas as pd
import pickle
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score
)
from scipy.sparse import hstack

# XGBoost - install if missing: pip install xgboost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not installed. Run: pip install xgboost")
    XGBOOST_AVAILABLE = False

print("=" * 60)
print("   AI FAKE REVIEW DETECTOR - MULTI-MODEL TRAINING")
print("=" * 60)

# -------------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------------
print("\n[1/6] Loading dataset...")
df = pd.read_csv("master_fake_review_dataset.csv")
df.dropna(subset=["review_text", "label"], inplace=True)
print(f"    Total samples: {len(df)}")
print(f"    Label distribution:\n{df['label'].value_counts()}")

# -------------------------------------------------------
# 2. Features and Labels
# -------------------------------------------------------
print("\n[2/6] Preparing features...")
X_text  = df["review_text"]
X_rating = df[["rating"]]
y = df["label"].astype(int)

# -------------------------------------------------------
# 3. TF-IDF Vectorization
# -------------------------------------------------------
print("\n[3/6] Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_text_vec = vectorizer.fit_transform(X_text)
X_final = hstack([X_text_vec, X_rating])
print(f"    Feature matrix shape: {X_final.shape}")

# -------------------------------------------------------
# 4. Train / Test Split
# -------------------------------------------------------
print("\n[4/6] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# -------------------------------------------------------
# 5. Define All Models
# -------------------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, C=1.0
    ),
    "SVM (LinearSVC)": LinearSVC(
        max_iter=2000, random_state=42, C=1.0
    ),
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

# -------------------------------------------------------
# 6. Train, Evaluate, Save Each Model
# -------------------------------------------------------
print("\n[5/6] Training and evaluating all models...\n")

all_metrics = {}
best_model_name = None
best_f1 = 0.0

for name, clf in models.items():
    print(f"  Training: {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted")

    # ROC-AUC (needs probability; LinearSVC uses decision_function)
    try:
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:
            y_prob = clf.decision_function(X_test)
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None

    all_metrics[name] = {
        "accuracy":  round(acc * 100, 2),
        "f1_score":  round(f1 * 100, 2),
        "precision": round(precision * 100, 2),
        "recall":    round(recall * 100, 2),
        "roc_auc":   round(auc * 100, 2) if auc is not None else "N/A"
    }

    print(f"    Accuracy : {acc*100:.2f}%")
    print(f"    F1-Score : {f1*100:.2f}%")
    print(f"    Precision: {precision*100:.2f}%")
    print(f"    Recall   : {recall*100:.2f}%")
    if auc:
        print(f"    ROC-AUC  : {auc*100:.2f}%")
    print()

    # Track best model
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

    # Save model with safe filename
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    pickle.dump(clf, open(f"model_{safe_name}.pkl", "wb"))
    print(f"    Saved: model_{safe_name}.pkl")

# -------------------------------------------------------
# 7. Save Vectorizer + Metrics + Best Model Info
# -------------------------------------------------------
print("\n[6/6] Saving vectorizer and metrics...")
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

all_metrics["_best_model"] = best_model_name
with open("model_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"\n  Best Model : {best_model_name}")
print(f"  Best F1    : {best_f1 * 100:.2f}%")
print("\n  Files saved:")
print("    - model_random_forest.pkl")
print("    - model_logistic_regression.pkl")
print("    - model_svm_linearsvc.pkl")
if XGBOOST_AVAILABLE:
    print("    - model_xgboost.pkl")
print("    - tfidf_vectorizer.pkl")
print("    - model_metrics.json")
print("\n  Use model_metrics.json in your Streamlit app")
print("  to display the comparison dashboard.")
print("=" * 60)

