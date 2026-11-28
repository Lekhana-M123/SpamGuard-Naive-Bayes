
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
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "dataset" / "processed"
RESULT_DIR = ROOT / "results"
GRAPH_DIR = ROOT / "graphs"

RESULT_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)

# -----------------------------------------
# 1. Load training and validation datasets
# -----------------------------------------

train_df = pd.read_csv(DATA_DIR / "train_fit.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train_df["message"].fillna("")
y_train = train_df["label"]

X_val = val_df["message"].fillna("")
y_val = val_df["label"]

# -----------------------------------------
# 2. Text vectorization
# -----------------------------------------

vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# -----------------------------------------
# 3. Train Naive Bayes model
# -----------------------------------------

model = MultinomialNB(alpha=0.1)

model.fit(X_train_vec, y_train)

# Get spam probabilities
spam_index = list(model.classes_).index("spam")

spam_probabilities = model.predict_proba(
    X_val_vec
)[:, spam_index]

# -----------------------------------------
# 4. Test different thresholds
# -----------------------------------------

thresholds = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]

results = []

y_true = (y_val == "spam").astype(int).to_numpy()

for threshold in thresholds:

    predictions = (
        spam_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true, predictions
    )

    precision = precision_score(
        y_true, predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true, predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true, predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp
    })

# -----------------------------------------
# 5. Display comparison
# -----------------------------------------

results_df = pd.DataFrame(results)

print("\n========== THRESHOLD ANALYSIS ==========")

print(
    results_df.round(4).to_string(index=False)
)

# -----------------------------------------
# 6. Save results
# -----------------------------------------

results_df.to_csv(
    RESULT_DIR / "threshold_analysis.csv",
    index=False
)

# -----------------------------------------
# 7. Generate precision-recall graph
# -----------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    results_df["threshold"],
    results_df["precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["threshold"],
    results_df["recall"],
    marker="s",
    label="Recall"
)

plt.plot(
    results_df["threshold"],
    results_df["f1_score"],
    marker="^",
    label="F1-score"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Evaluation Score")

plt.title(
    "Effect of Decision Threshold on Classification"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "threshold_comparison.png",
    dpi=300
)

plt.show()

print("\nResults saved successfully.")
print("Threshold analysis completed.")

