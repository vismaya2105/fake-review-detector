# Fake Review Detection Model

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from scipy.sparse import hstack

# -----------------------------
# 1 Load Dataset
# -----------------------------

df = pd.read_csv("master_fake_review_dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())

# -----------------------------
# 2 Separate Features and Label
# -----------------------------

X_text = df["review_text"]
X_rating = df[["rating"]]
y = df["label"]

# -----------------------------
# 3 Convert Text to TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer(max_features=5000)

X_text_vectorized = vectorizer.fit_transform(X_text)

# -----------------------------
# 4 Combine Text + Rating
# -----------------------------

X_final = hstack([X_text_vectorized, X_rating])

# -----------------------------
# 5 Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
print("Starting Model Training...")

# -----------------------------
# 6 Train Model
# -----------------------------

model = RandomForestClassifier(n_estimators=200)

model.fit(X_train, y_train)

print("Model Training Completed")

# 7 Model Evaluation
# -----------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 8 Save Model and Vectorizer
# -----------------------------

pickle.dump(model, open("fake_review_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("Model and Vectorizer Saved Successfully")

# -----------------------------
# 9 Test Prediction
# -----------------------------

test_review = ["Amazing product highly recommended"]

test_vec = vectorizer.transform(test_review)

test_final = hstack([test_vec, [[5]]])

prediction = model.predict(test_final)

if prediction[0] == 1:
    print("Prediction: Genuine Review")
else:
    print("Prediction: Fake Review")



