
"""Canonical independent-test evaluation for the fixed final model settings."""

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# -----------------------------------------
# 1. Project paths
# -----------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "dataset" / "processed"
RESULT_DIR = ROOT / "results"
GRAPH_DIR = ROOT / "graphs"

RESULT_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)

# -----------------------------------------
# 2. Load datasets
# -----------------------------------------

# This script alone writes the final_test_* metrics/predictions and final-test
# confusion-matrix figure. Test labels are used only for the final evaluation.
train_df = pd.read_csv(DATA_DIR / "train_fit.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

# Combine training and validation data
final_train = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

X_train = final_train["message"].fillna("")
y_train = final_train["label"]

X_test = test_df["message"].fillna("")
y_test = (test_df["label"] == "spam").astype(int)

print("\n========== FINAL DATASET INFORMATION ==========")

print("Final training records:", len(final_train))
print("Independent test records:", len(test_df))

# -----------------------------------------
# 3. Feature extraction
# -----------------------------------------

vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Vocabulary size:", len(vectorizer.vocabulary_))

# -----------------------------------------
# 4. Train final model
# -----------------------------------------

ALPHA = 0.1
THRESHOLD = 0.3

model = MultinomialNB(alpha=ALPHA)

model.fit(X_train_vec, y_train)

print("\nFinal model training completed.")

# -----------------------------------------
# 5. Predict test data
# -----------------------------------------

spam_index = list(model.classes_).index("spam")

probabilities = model.predict_proba(
    X_test_vec
)[:, spam_index]

predictions = (
    probabilities >= THRESHOLD
).astype(int)

# -----------------------------------------
# 6. Calculate final metrics
# -----------------------------------------

metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(
        y_test, predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, predictions, zero_division=0
    ),
    "f1_score": f1_score(
        y_test, predictions, zero_division=0
    ),
    "roc_auc": roc_auc_score(y_test, probabilities),
    "pr_auc": average_precision_score(
        y_test, probabilities
    )
}

print("\n========== FINAL TEST RESULTS ==========")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# -----------------------------------------
# 7. Confusion matrix
# -----------------------------------------

cm = confusion_matrix(y_test, predictions)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Ham", "Spam"],
        zero_division=0
    )
)

# -----------------------------------------
# 8. Save predictions
# -----------------------------------------

prediction_df = pd.DataFrame({
    "message": X_test,
    "actual_label": test_df["label"],
    "predicted_label": [
        "spam" if value == 1 else "ham"
        for value in predictions
    ],
    "spam_probability": probabilities,
    "correct_prediction": (
        (y_test.to_numpy() == predictions)
    )
})

prediction_df.to_csv(
    RESULT_DIR / "final_test_predictions.csv",
    index=False
)

pd.DataFrame(
    list(metrics.items()),
    columns=["metric", "value"]
).to_csv(
    RESULT_DIR / "final_test_metrics.csv",
    index=False
)

# -----------------------------------------
# 9. Plot confusion matrix
# -----------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Final Test Dataset Confusion Matrix")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "final_test_confusion_matrix.png",
    dpi=300
)

plt.show()

# -----------------------------------------
# 10. Completion
# -----------------------------------------

print("\n========== OUTPUT FILES ==========")

print(RESULT_DIR / "final_test_predictions.csv")
print(RESULT_DIR / "final_test_metrics.csv")
print(GRAPH_DIR / "final_test_confusion_matrix.png")

print("\nFinal independent test evaluation completed.")
