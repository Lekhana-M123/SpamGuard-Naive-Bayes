
import pandas as pd
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Project paths
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "dataset" / "processed"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Configuration selected through validation
ALPHA = 0.1
THRESHOLD = 0.3

# Load training and validation datasets
train_df = pd.read_csv(DATA_DIR / "train_fit.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

# Combine training and validation data
final_train = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

X_train = final_train["message"].fillna("")
y_train = final_train["label"]

print("Training records:", len(final_train))

# Fit vectorizer
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)

# Train final model
model = MultinomialNB(alpha=ALPHA)

model.fit(X_train_vec, y_train)

# Save model
joblib.dump(
    model,
    MODEL_DIR / "naive_bayes_model.pkl"
)

# Save vectorizer
joblib.dump(
    vectorizer,
    MODEL_DIR / "count_vectorizer.pkl"
)

# Save configuration
configuration = {
    "algorithm": "Multinomial Naive Bayes",
    "alpha": ALPHA,
    "decision_threshold": THRESHOLD,
    "training_records": len(final_train),
    "vocabulary_size": len(vectorizer.vocabulary_)
}

joblib.dump(
    configuration,
    MODEL_DIR / "model_config.pkl"
)

print("\n========== MODEL SAVING ==========")

print("Model saved successfully.")
print("Vectorizer saved successfully.")
print("Configuration saved successfully.")

print("\nSaved files:")

print(MODEL_DIR / "naive_bayes_model.pkl")
print(MODEL_DIR / "count_vectorizer.pkl")
print(MODEL_DIR / "model_config.pkl")
