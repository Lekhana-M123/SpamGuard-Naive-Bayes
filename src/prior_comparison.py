
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "dataset" / "processed"
RESULT_DIR = ROOT / "results"

RESULT_DIR.mkdir(exist_ok=True)

# Load datasets
train_df = pd.read_csv(DATA_DIR / "train_fit.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train_df["message"].fillna("")
y_train = train_df["label"]

X_val = val_df["message"].fillna("")
y_val = val_df["label"]

# Vectorization
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Actual training priors
actual_priors = (
    y_train.value_counts(normalize=True)
    .reindex(["ham", "spam"])
    .values
)

# Equal priors
equal_priors = [0.5, 0.5]

# Train both models
models = {
    "Actual Priors": MultinomialNB(
        alpha=0.1,
        class_prior=actual_priors
    ),
    "Equal Priors": MultinomialNB(
        alpha=0.1,
        class_prior=equal_priors
    )
}

results = []

print("\n===== PRIOR PROBABILITY COMPARISON =====")

for name, model in models.items():

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_val_vec)

    probabilities = model.predict_proba(X_val_vec)

    spam_index = list(model.classes_).index("spam")

    spam_probs = probabilities[:, spam_index]

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions,
            pos_label="spam",
            zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions,
            pos_label="spam",
            zero_division=0
        ),
        "f1_score": f1_score(
            y_val, predictions,
            pos_label="spam",
            zero_division=0
        ),
        "pr_auc": average_precision_score(
            (y_val == "spam").astype(int),
            spam_probs
        )
    })

    print(f"\n{name}")

    print("Confusion matrix:")
    print(confusion_matrix(
        y_val,
        predictions,
        labels=["ham", "spam"]
    ))

# Display comparison
results_df = pd.DataFrame(results)

print("\n===== COMPARISON RESULTS =====")

print(results_df.round(4).to_string(index=False))

# Verify word likelihoods are identical
model_a = models["Actual Priors"]
model_b = models["Equal Priors"]

assert (
    model_a.feature_log_prob_
    == model_b.feature_log_prob_
).all()

print("\nWord likelihoods are identical.")

results_df.to_csv(
    RESULT_DIR / "prior_comparison.csv",
    index=False
)

print("\nResults saved successfully.")
