import pandas as pd
import pickle
import json
import numpy as np

from sklearn.model_selection  import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from scipy.sparse import hstack

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not installed. Run: pip install xgboost")
    XGBOOST_AVAILABLE = False

df = pd.read_csv("master_fake_review_dataset.csv")
df.dropna(subset=["review_text", "label"], inplace=True)
print(f"Total samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")

X_text   = df["review_text"]
X_rating = df[["rating"]]
y        = df["label"].astype(int)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_text_vec = vectorizer.fit_transform(X_text)
X_final    = hstack([X_text_vec, X_rating])
print(f"Feature matrix shape: {X_final.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
 
rf_clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
 
# Evaluation
acc       = accuracy_score(y_test, y_pred_rf)
f1        = f1_score(y_test, y_pred_rf, average='weighted')
precision = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
recall    = recall_score(y_test, y_pred_rf, average='weighted')
y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]
auc       = roc_auc_score(y_test, y_prob_rf)
 
# Save model
pickle.dump(rf_clf, open("model_random_forest.pkl", "wb"))

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
 
lr_clf = LogisticRegression(
    max_iter=1000,
    random_state=42,
    C=1.0
)
lr_clf.fit(X_train, y_train)
y_pred_lr = lr_clf.predict(X_test)
 
# Evaluation
acc       = accuracy_score(y_test, y_pred_lr)
f1        = f1_score(y_test, y_pred_lr, average='weighted')
precision = precision_score(y_test, y_pred_lr, average='weighted', zero_division=0)
recall    = recall_score(y_test, y_pred_lr, average='weighted')
y_prob_lr = lr_clf.predict_proba(X_test)[:, 1]
auc       = roc_auc_score(y_test, y_prob_lr)
 
# Save model
pickle.dump(lr_clf, open("model_logistic_regression.pkl", "wb"))

# SVM
from sklearn.svm import LinearSVC
 
svm_clf = LinearSVC(
    max_iter=2000,
    random_state=42,
    C=1.0
)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
 
# Evaluation
acc       = accuracy_score(y_test, y_pred_svm)
f1        = f1_score(y_test, y_pred_svm, average='weighted')
precision = precision_score(y_test, y_pred_svm, average='weighted', zero_division=0)
recall    = recall_score(y_test, y_pred_svm, average='weighted')
# LinearSVC uses decision_function instead of predict_proba
y_prob_svm = svm_clf.decision_function(X_test)
auc        = roc_auc_score(y_test, y_prob_svm)
 
# Save model
pickle.dump(svm_clf, open("model_svm_linearsvc.pkl", "wb"))

#Installation Check
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not installed. Run: pip install xgboost")
    XGBOOST_AVAILABLE = False

#XGBoost
from xgboost import XGBClassifier
 
xgb_clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
 
# Evaluation
acc       = accuracy_score(y_test, y_pred_xgb)
f1        = f1_score(y_test, y_pred_xgb, average='weighted')
precision = precision_score(y_test, y_pred_xgb, average='weighted', zero_division=0)
recall    = recall_score(y_test, y_pred_xgb, average='weighted')
y_prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]
auc        = roc_auc_score(y_test, y_prob_xgb)
 
# Save model
pickle.dump(xgb_clf, open("model_xgboost.pkl", "wb"))





