import os
import gdown
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score
)
from scipy.sparse import hstack
import pickle
import json

# -------------------------------------------------------
# 1. Download CSV from Google Drive
# -------------------------------------------------------
if not os.path.exists("master_fake_review_dataset.csv"):
    print("Downloading dataset from Google Drive...")
    url = "https://drive.google.com/uc?id=1t-QvksSsDCqFzFSrf0dIEIPCbmxdVgYL"
    gdown.download(url, "master_fake_review_dataset.csv", quiet=False)
    print("Dataset downloaded successfully!")

# -------------------------------------------------------
# 2. Train Models only if not already trained
# -------------------------------------------------------
if not os.path.exists("tfidf_vectorizer.pkl"):
    print("Starting model training...")

    # Load dataset
    df = pd.read_csv("master_fake_review_dataset.csv")
    df.dropna(subset=["review_text", "label"], inplace=True)
    print(f"Dataset loaded: {len(df)} samples")

    # Features and labels
    X_text   = df["review_text"]
    X_rating = df[["rating"]]
    y        = df["label"].astype(int)

    # TF-IDF Vectorization
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_text_vec = vectorizer.fit_transform(X_text)
    X_final    = hstack([X_text_vec, X_rating])
    print(f"Feature matrix shape: {X_final.shape}")

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Define all 4 models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0
        ),
        "SVM (LinearSVC)": LinearSVC(
            max_iter=2000,
            random_state=42,
            C=1.0
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    # Train, evaluate and save each model
    all_metrics    = {}
    best_model_name = None
    best_f1        = 0.0

    for name, clf in models.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc       = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_test, y_pred, average="weighted")

        all_metrics[name] = {
            "accuracy":  round(acc * 100, 2),
            "f1_score":  round(f1 * 100, 2),
            "precision": round(precision * 100, 2),
            "recall":    round(recall * 100, 2),
        }

        print(f"  {name} -> Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")

        # Track best model
        if f1 > best_f1:
            best_f1         = f1
            best_model_name = name

        # Save model
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        pickle.dump(clf, open(f"model_{safe_name}.pkl", "wb"))
        print(f"  Saved: model_{safe_name}.pkl")

    # Save vectorizer
    pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))
    print("Saved: tfidf_vectorizer.pkl")

    # Save metrics JSON
    all_metrics["_best_model"] = best_model_name
    with open("model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Saved: model_metrics.json")

    print("=" * 50)
    print(f"Training Complete!")
    print(f"Best Model : {best_model_name}")
    print(f"Best F1    : {best_f1 * 100:.2f}%")
    print("=" * 50)

else:
    print("Models already trained. Skipping training.")
