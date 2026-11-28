
"""Full-dataset prior illustration; not a held-out evaluation or model-selection run.

This demonstration intentionally fits on the complete UCI collection, including
the reserved independent-test partition. Do not use its probabilities as
predictive-performance evidence.
"""

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------------------
# 1. Load the original UCI dataset
# -----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "SMSSpamCollection"
)

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    quoting=3,
    encoding="utf-8"
)

print("\n========== PRIOR PROBABILITY ANALYSIS ==========")
print(
    "ILLUSTRATIVE ONLY: fits on the full dataset, including the independent-test "
    "partition; not for performance reporting or model selection."
)

# -----------------------------------------
# 2. Calculate actual class priors
# -----------------------------------------

class_priors = (
    df["label"]
    .value_counts(normalize=True)
    .sort_index()
)

print("\nActual class distribution:")

for label, probability in class_priors.items():
    print(f"P({label}) = {probability:.4f}")

# -----------------------------------------
# 3. Convert messages into word counts
# -----------------------------------------

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["message"])

y = df["label"]

# -----------------------------------------
# 4. Train model using actual priors
# -----------------------------------------

actual_prior_values = class_priors.values

model_actual = MultinomialNB(
    alpha=1,
    class_prior=actual_prior_values
)

model_actual.fit(X, y)

# -----------------------------------------
# 5. Train model using equal priors
# -----------------------------------------

equal_prior_values = [0.5, 0.5]

model_equal = MultinomialNB(
    alpha=1,
    class_prior=equal_prior_values
)

model_equal.fit(X, y)

# Verify that word likelihoods are unchanged
assert np.allclose(
    model_actual.feature_log_prob_,
    model_equal.feature_log_prob_
)

print("\nWord likelihoods remain identical.")
print("Only class priors have changed.")

# -----------------------------------------
# 6. Test selected example messages
# -----------------------------------------

test_messages = [
    "Congratulations you won a free prize",
    "Are we meeting tomorrow",
    "Free entry claim your prize now",
    "Please call me when you reach home"
]

X_test = vectorizer.transform(test_messages)

actual_predictions = model_actual.predict_proba(X_test)

equal_predictions = model_equal.predict_proba(X_test)

# -----------------------------------------
# 7. Compare posterior probabilities
# -----------------------------------------

spam_index = list(model_actual.classes_).index("spam")

print("\n========== PREDICTION COMPARISON ==========")

for i, message in enumerate(test_messages):

    actual_spam = actual_predictions[i][spam_index] * 100

    equal_spam = equal_predictions[i][spam_index] * 100

    print("\nMessage:", message)

    print(
        f"Spam probability (actual priors): "
        f"{actual_spam:.2f}%"
    )

    print(
        f"Spam probability (equal priors): "
        f"{equal_spam:.2f}%"
    )

    print(
        f"Change: {equal_spam - actual_spam:+.2f} "
        "percentage points"
    )

print("\nExperiment completed successfully.")
