
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
    average_precision_score,
    roc_auc_score
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
# 2. Load training and validation data
# -----------------------------------------

train_df = pd.read_csv(DATA_DIR / "train_fit.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train_df["message"].fillna("")
y_train = train_df["label"]

X_val = val_df["message"].fillna("")
y_val = val_df["label"]

print("\n========== ALPHA SENSITIVITY ANALYSIS ==========")

print("Training records:", len(X_train))
print("Validation records:", len(X_val))

# -----------------------------------------
# 3. Convert messages into word frequencies
# -----------------------------------------

vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)

X_val_vec = vectorizer.transform(X_val)

print("Vocabulary size:", len(vectorizer.vocabulary_))

# -----------------------------------------
# 4. Test different alpha values
# -----------------------------------------

alpha_values = [0.01, 0.1, 0.5, 1, 2, 5, 10]

results = []

for alpha in alpha_values:

    print(f"\nTesting alpha = {alpha}")

    model = MultinomialNB(alpha=alpha)

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_val_vec)

    probabilities = model.predict_proba(X_val_vec)

    spam_index = list(model.classes_).index("spam")

    spam_probabilities = probabilities[:, spam_index]

    # Calculate evaluation metrics

    accuracy = accuracy_score(y_val, predictions)

    precision = precision_score(
        y_val, predictions,
        pos_label="spam",
        zero_division=0
    )

    recall = recall_score(
        y_val, predictions,
        pos_label="spam",
        zero_division=0
    )

    f1 = f1_score(
        y_val, predictions,
        pos_label="spam",
        zero_division=0
    )

    pr_auc = average_precision_score(
        (y_val == "spam").astype(int),
        spam_probabilities
    )

    roc_auc = roc_auc_score(
        (y_val == "spam").astype(int),
        spam_probabilities
    )

    results.append({
        "alpha": alpha,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc
    })

# -----------------------------------------
# 5. Display comparison table
# -----------------------------------------

results_df = pd.DataFrame(results)

print("\n========== ALPHA COMPARISON RESULTS ==========")

print(
    results_df.round(4).to_string(index=False)
)

# -----------------------------------------
# 6. Identify best alpha using validation PR-AUC
# -----------------------------------------

best_index = results_df["pr_auc"].idxmax()

best_result = results_df.loc[best_index]

print("\n========== BEST VALIDATION RESULT ==========")

print("Best alpha:", best_result["alpha"])

print(f"PR-AUC: {best_result['pr_auc']:.4f}")

print(f"F1-score: {best_result['f1_score']:.4f}")

print(f"Precision: {best_result['precision']:.4f}")

print(f"Recall: {best_result['recall']:.4f}")

# -----------------------------------------
# 7. Save experimental results
# -----------------------------------------

results_path = RESULT_DIR / "alpha_comparison.csv"

results_df.to_csv(results_path, index=False)

print("\nResults saved:", results_path)

# -----------------------------------------
# 8. Generate comparison graph
# -----------------------------------------

plt.figure(figsize=(11, 6))

plt.plot(
    results_df["alpha"],
    results_df["f1_score"],
    marker="o",
    label="F1-score"
)

plt.plot(
    results_df["alpha"],
    results_df["pr_auc"],
    marker="s",
    label="PR-AUC"
)

plt.plot(
    results_df["alpha"],
    results_df["recall"],
    marker="^",
    label="Recall"
)

plt.xscale("log")

plt.xlabel("Smoothing Parameter (Alpha)")

plt.ylabel("Evaluation Score")

plt.title(
    "Effect of Laplace Smoothing on Model Performance"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

graph_path = GRAPH_DIR / "alpha_comparison.png"

plt.savefig(graph_path, dpi=300)

plt.show()

print("\nGraph saved:", graph_path)

print("\nAlpha sensitivity experiment completed.")
