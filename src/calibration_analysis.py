
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    brier_score_loss,
    log_loss
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
y_val = (val_df["label"] == "spam").astype(int)

print("\n========== DATASET INFORMATION ==========")

print("Training records:", len(X_train))
print("Validation records:", len(X_val))

# -----------------------------------------
# 3. Text vectorization
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

spam_index = list(model.classes_).index("spam")

probabilities = model.predict_proba(
    X_val_vec
)[:, spam_index]

# -----------------------------------------
# 5. Calculate calibration metrics
# -----------------------------------------

brier = brier_score_loss(
    y_val,
    probabilities
)

logloss = log_loss(
    y_val,
    probabilities,
    labels=[0, 1]
)

print("\n========== CALIBRATION ANALYSIS ==========")

print(f"Brier Score: {brier:.5f}")
print(f"Log Loss:    {logloss:.5f}")

# -----------------------------------------
# 6. Reliability analysis with bin counts
# -----------------------------------------

bin_edges = np.linspace(0, 1, 11)

bin_ids = np.searchsorted(
    bin_edges[1:-1],
    probabilities
)

bin_results = []

ece = 0.0
total = len(y_val)

for i in range(10):

    mask = bin_ids == i

    count = int(mask.sum())

    if count == 0:
        continue

    mean_predicted = probabilities[mask].mean()

    actual_frequency = y_val.to_numpy()[mask].mean()

    calibration_error = abs(
        mean_predicted - actual_frequency
    )

    ece += (count / total) * calibration_error

    bin_results.append({
        "bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
        "count": count,
        "mean_predicted": mean_predicted,
        "actual_frequency": actual_frequency,
        "calibration_error": calibration_error
    })

calibration_results = pd.DataFrame(bin_results)

print("\n========== RELIABILITY RESULTS ==========")

print(
    calibration_results.round(4).to_string(index=False)
)

print(f"\nExpected Calibration Error (ECE): {ece:.5f}")

# -----------------------------------------
# 7. Save calibration results
# -----------------------------------------

calibration_results.to_csv(
    RESULT_DIR / "calibration_results.csv",
    index=False
)

metrics_df = pd.DataFrame([{
    "brier_score": brier,
    "log_loss": logloss,
    "expected_calibration_error": ece
}])

metrics_df.to_csv(
    RESULT_DIR / "calibration_metrics.csv",
    index=False
)

# -----------------------------------------
# 8. Generate improved reliability diagram
# -----------------------------------------

plt.figure(figsize=(9, 7))

plt.plot(
    calibration_results["mean_predicted"],
    calibration_results["actual_frequency"],
    marker="o",
    label="Naive Bayes"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Spam Probability")

plt.ylabel("Actual Spam Frequency")

plt.title("Naive Bayes Probability Calibration")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

graph_path = GRAPH_DIR / "calibration_curve.png"

plt.savefig(
    graph_path,
    dpi=300
)

plt.show()

print("\n========== OUTPUT FILES ==========")

print("Calibration results:", RESULT_DIR / "calibration_results.csv")
print("Calibration metrics:", RESULT_DIR / "calibration_metrics.csv")
print("Reliability graph:", graph_path)

print("\nCalibration analysis completed successfully.")
