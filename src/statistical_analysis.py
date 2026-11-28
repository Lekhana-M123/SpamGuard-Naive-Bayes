
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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

train_df = pd.read_csv(DATA_DIR / "train_fit.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train_df["message"].fillna("")
y_train = train_df["label"]

X_val = val_df["message"].fillna("")
y_val = (val_df["label"] == "spam").astype(int).to_numpy()

print("\n========== DATASET INFORMATION ==========")

print("Training records:", len(train_df))
print("Validation records:", len(val_df))

# -----------------------------------------
# 3. Vectorization
# -----------------------------------------

vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

print("Vocabulary size:", len(vectorizer.vocabulary_))

# -----------------------------------------
# 4. Train Naive Bayes model
# -----------------------------------------

model = MultinomialNB(alpha=0.1)

model.fit(X_train_vec, y_train)

predicted_probabilities = model.predict_proba(X_val_vec)

spam_index = list(model.classes_).index("spam")

spam_probabilities = predicted_probabilities[:, spam_index]

# Use the baseline decision threshold of 0.5
y_pred = (spam_probabilities >= 0.5).astype(int)

# -----------------------------------------
# 5. Original evaluation metrics
# -----------------------------------------

original_metrics = {
    "accuracy": accuracy_score(y_val, y_pred),
    "precision": precision_score(
        y_val, y_pred, zero_division=0
    ),
    "recall": recall_score(
        y_val, y_pred, zero_division=0
    ),
    "f1_score": f1_score(
        y_val, y_pred, zero_division=0
    )
}

print("\n========== ORIGINAL METRICS ==========")

for metric, value in original_metrics.items():
    print(f"{metric}: {value:.4f}")

# -----------------------------------------
# 6. Bootstrap resampling
# -----------------------------------------

BOOTSTRAP_ITERATIONS = 1000

rng = np.random.default_rng(42)

grouped_validation = val_df.assign(
    message=X_val.to_numpy()
).groupby("message", sort=False)
if (grouped_validation["label"].nunique() > 1).any():
    raise ValueError("Identical validation messages have conflicting labels.")

message_groups = list(grouped_validation.indices.values())
n_groups = len(message_groups)

bootstrap_results = []

print("\n========== BOOTSTRAP ANALYSIS ==========")

print("Bootstrap iterations:", BOOTSTRAP_ITERATIONS)
print("Unique validation message groups:", n_groups)
print("Sampling unit: identical message-text group")

for iteration in range(BOOTSTRAP_ITERATIONS):

    # Resample unique message groups so duplicate rows stay together.
    sampled_groups = rng.choice(
        n_groups,
        size=n_groups,
        replace=True
    )
    sample_indices = np.concatenate(
        [message_groups[group_index] for group_index in sampled_groups]
    )

    y_true_sample = y_val[sample_indices]
    y_pred_sample = y_pred[sample_indices]

    # Calculate evaluation metrics
    accuracy = accuracy_score(
        y_true_sample,
        y_pred_sample
    )

    precision = precision_score(
        y_true_sample,
        y_pred_sample,
        zero_division=0
    )

    recall = recall_score(
        y_true_sample,
        y_pred_sample,
        zero_division=0
    )

    f1 = f1_score(
        y_true_sample,
        y_pred_sample,
        zero_division=0
    )

    bootstrap_results.append({
        "iteration": iteration + 1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    if (iteration + 1) % 100 == 0:
        print(
            f"Completed {iteration + 1} iterations"
        )

# -----------------------------------------
# 7. Save bootstrap results
# -----------------------------------------

bootstrap_df = pd.DataFrame(bootstrap_results)

bootstrap_df.to_csv(
    RESULT_DIR / "bootstrap_results.csv",
    index=False
)

# -----------------------------------------
# 8. Calculate 95% confidence intervals
# -----------------------------------------

confidence_results = []

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1_score"
]

for metric in metrics:

    values = bootstrap_df[metric]

    lower = np.percentile(values, 2.5)
    upper = np.percentile(values, 97.5)

    mean_value = values.mean()
    std_value = values.std(ddof=1)

    confidence_results.append({
        "metric": metric,
        "original_score": original_metrics[metric],
        "bootstrap_mean": mean_value,
        "standard_deviation": std_value,
        "lower_95_CI": lower,
        "upper_95_CI": upper
    })

confidence_df = pd.DataFrame(confidence_results)

print("\n========== 95% CONFIDENCE INTERVALS ==========")

print(
    confidence_df.round(4).to_string(index=False)
)

confidence_df.to_csv(
    RESULT_DIR / "confidence_intervals.csv",
    index=False
)

# -----------------------------------------
# 9. Generate F1-score distribution graph
# -----------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    bootstrap_df["f1_score"],
    bins=30,
    edgecolor="black"
)

plt.axvline(
    original_metrics["f1_score"],
    linestyle="--",
    linewidth=2,
    label="Original F1-score"
)

f1_lower = confidence_df.loc[
    confidence_df["metric"] == "f1_score",
    "lower_95_CI"
].iloc[0]

f1_upper = confidence_df.loc[
    confidence_df["metric"] == "f1_score",
    "upper_95_CI"
].iloc[0]

plt.axvline(
    f1_lower,
    linestyle=":",
    label="Lower 95% CI"
)

plt.axvline(
    f1_upper,
    linestyle=":",
    label="Upper 95% CI"
)

plt.xlabel("F1-score")
plt.ylabel("Frequency")

plt.title(
    "Validation F1-score: Message-Group Bootstrap"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "bootstrap_f1_distribution.png",
    dpi=300
)

plt.show()

# -----------------------------------------
# 10. Final output
# -----------------------------------------

print("\n========== OUTPUT FILES ==========")

print(
    "Bootstrap results:",
    RESULT_DIR / "bootstrap_results.csv"
)

print(
    "Confidence intervals:",
    RESULT_DIR / "confidence_intervals.csv"
)

print(
    "F1 distribution graph:",
    GRAPH_DIR / "bootstrap_f1_distribution.png"
)

print("\nStatistical uncertainty analysis completed.")
