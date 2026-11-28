
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

# -----------------------------------------
# 1. Project paths
# -----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
RESULT_DIR = PROJECT_ROOT / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Historical alpha=1.0 baseline only. Do not use these test results for
# model or threshold selection; final_test_evaluation.py owns final_test_* outputs.
BASELINE_ALPHA = 1.0
BASELINE_THRESHOLD = 0.5

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

# -----------------------------------------
# 2. Load training and testing datasets
# -----------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["message"].fillna("")
y_train = train_df["label"]

X_test = test_df["message"].fillna("")
y_test = test_df["label"]

print("\n========== MODEL TRAINING ==========")
print(
    "HISTORICAL BASELINE ONLY: alpha=1.0, default 0.5 decision threshold; "
    "not the canonical final evaluation."
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# -----------------------------------------
# 3. Convert text into word frequencies
# -----------------------------------------

vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)

X_test_vectorized = vectorizer.transform(X_test)

print("\nVocabulary size:", len(vectorizer.vocabulary_))

# -----------------------------------------
# 4. Train Multinomial Naive Bayes
# -----------------------------------------

model = MultinomialNB(alpha=BASELINE_ALPHA)

model.fit(X_train_vectorized, y_train)

print("\nModel training completed.")

# -----------------------------------------
# 5. Generate predictions
# -----------------------------------------

# This baseline uses sklearn's default 0.5 decision rule.
y_pred = model.predict(X_test_vectorized)

y_prob = model.predict_proba(X_test_vectorized)

spam_index = list(model.classes_).index("spam")

spam_probabilities = y_prob[:, spam_index]

# -----------------------------------------
# 6. Calculate evaluation metrics
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    pos_label="spam",
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    pos_label="spam",
    zero_division=0
)

f1 = f1_score(
    y_test, y_pred,
    pos_label="spam",
    zero_division=0
)

roc_auc = roc_auc_score(
    (y_test == "spam").astype(int),
    spam_probabilities
)

pr_auc = average_precision_score(
    (y_test == "spam").astype(int),
    spam_probabilities
)

# -----------------------------------------
# 7. Display evaluation results
# -----------------------------------------

print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

# -----------------------------------------
# 8. Confusion matrix
# -----------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["ham", "spam"]
)

print("\n========== CONFUSION MATRIX ==========")

print("                 Predicted")
print("                 Ham    Spam")
print(f"Actual Ham       {cm[0][0]:<6} {cm[0][1]}")
print(f"Actual Spam      {cm[1][0]:<6} {cm[1][1]}")

# -----------------------------------------
# 9. Detailed classification report
# -----------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["ham", "spam"],
        zero_division=0
    )
)

# -----------------------------------------
# 10. Save predictions
# -----------------------------------------

prediction_df = pd.DataFrame({
    "message": X_test,
    "actual_label": y_test,
    "predicted_label": y_pred,
    "spam_probability": spam_probabilities,
    "evaluation_role": "historical_baseline",
    "alpha": BASELINE_ALPHA,
    "decision_threshold": BASELINE_THRESHOLD,
})

prediction_path = RESULT_DIR / "predictions.csv"
metrics_path = RESULT_DIR / "model_metrics.csv"
canonical_output_paths = {
    RESULT_DIR / "final_test_predictions.csv",
    RESULT_DIR / "final_test_metrics.csv",
}
if prediction_path in canonical_output_paths or metrics_path in canonical_output_paths:
    raise RuntimeError("Historical baseline must not write canonical final-test files.")

prediction_df.to_csv(
    prediction_path,
    index=False
)

print("\nPredictions saved:", prediction_path)

# -----------------------------------------
# 11. Save numerical metrics
# -----------------------------------------

metrics = pd.DataFrame([{
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "evaluation_role": "historical_baseline",
    "alpha": BASELINE_ALPHA,
    "decision_threshold": BASELINE_THRESHOLD,
}])

metrics.to_csv(
    metrics_path,
    index=False
)

print("Metrics saved:", metrics_path)
print(
    "Historical alpha=1.0 baseline only; do not use these results "
    "for final model or threshold selection."
)

print("\nModel evaluation completed successfully.")
