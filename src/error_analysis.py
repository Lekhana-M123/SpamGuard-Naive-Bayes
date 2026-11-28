
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix

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

y_true = (
    val_df["label"] == "spam"
).astype(int).to_numpy()

# -----------------------------------------
# 3. Vectorization
# -----------------------------------------

vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# -----------------------------------------
# 4. Train model
# -----------------------------------------

model = MultinomialNB(alpha=0.1)

model.fit(X_train_vec, y_train)

spam_index = list(model.classes_).index("spam")

probabilities = model.predict_proba(
    X_val_vec
)[:, spam_index]

y_pred = (probabilities >= 0.5).astype(int)

# -----------------------------------------
# 5. Identify prediction errors
# -----------------------------------------

error_df = pd.DataFrame({
    "message": X_val.to_numpy(),
    "actual": np.where(y_true == 1, "spam", "ham"),
    "predicted": np.where(y_pred == 1, "spam", "ham"),
    "spam_probability": probabilities
})

error_df["prediction_status"] = np.where(
    y_true == y_pred,
    "Correct",
    "Incorrect"
)

error_df["error_type"] = np.select(
    [
        (y_true == 0) & (y_pred == 1),
        (y_true == 1) & (y_pred == 0)
    ],
    [
        "False Positive",
        "False Negative"
    ],
    default="Correct"
)

errors = error_df[
    error_df["prediction_status"] == "Incorrect"
].copy()

print("\n========== ERROR ANALYSIS ==========")

print("Total validation records:", len(error_df))
print("Correct predictions:", len(error_df) - len(errors))
print("Incorrect predictions:", len(errors))

# -----------------------------------------
# 6. False positive and negative analysis
# -----------------------------------------

fp = errors[
    errors["error_type"] == "False Positive"
]

fn = errors[
    errors["error_type"] == "False Negative"
]

print("\n========== ERROR TYPES ==========")

print("False Positives:", len(fp))
print("False Negatives:", len(fn))

print("\nFalse Positive Examples:")

for message in fp["message"].head(5):
    print("-", message[:150])

print("\nFalse Negative Examples:")

for message in fn["message"].head(5):
    print("-", message[:150])

# -----------------------------------------
# 7. Prediction confidence
# -----------------------------------------

errors["prediction_confidence"] = np.where(
    errors["predicted"] == "spam",
    errors["spam_probability"],
    1 - errors["spam_probability"]
)

print("\n========== ERROR CONFIDENCE ==========")

if len(errors) > 0:

    print(
        "Mean error confidence:",
        round(errors["prediction_confidence"].mean(), 4)
    )

    print(
        "Maximum error confidence:",
        round(errors["prediction_confidence"].max(), 4)
    )

    print(
        "Minimum error confidence:",
        round(errors["prediction_confidence"].min(), 4)
    )

# -----------------------------------------
# 8. Save error records
# -----------------------------------------

errors.to_csv(
    RESULT_DIR / "error_analysis.csv",
    index=False
)

summary = pd.DataFrame({
    "category": [
        "Correct",
        "False Positive",
        "False Negative"
    ],
    "count": [
        len(error_df) - len(errors),
        len(fp),
        len(fn)
    ]
})

summary.to_csv(
    RESULT_DIR / "error_summary.csv",
    index=False
)

# -----------------------------------------
# 9. Error distribution graph
# -----------------------------------------

plt.figure(figsize=(8, 6))

plt.bar(
    ["Correct", "False Positive", "False Negative"],
    [
        len(error_df) - len(errors),
        len(fp),
        len(fn)
    ]
)

plt.xlabel("Prediction Category")
plt.ylabel("Number of Messages")

plt.title("Distribution of Classification Errors")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "error_distribution.png",
    dpi=300
)

plt.show()

# -----------------------------------------
# 10. Error confidence graph
# -----------------------------------------

if len(errors) > 0:

    plt.figure(figsize=(9, 6))

    plt.hist(
        errors["prediction_confidence"],
        bins=15,
        edgecolor="black"
    )

    plt.xlabel("Confidence of Incorrect Prediction")
    plt.ylabel("Number of Errors")

    plt.title("Confidence Distribution of Misclassified Messages")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "error_confidence.png",
        dpi=300
    )

    plt.show()

# -----------------------------------------
# 11. Most common words in errors
# -----------------------------------------

if len(errors) > 0:

    word_counter = Counter()

    for message in errors["message"]:

        words = str(message).lower().split()

        word_counter.update(words)

    print("\n========== COMMON ERROR WORDS ==========")

    for word, count in word_counter.most_common(15):

        print(f"{word}: {count}")

print("\nError analysis completed successfully.")

print("Results saved in:", RESULT_DIR)
print("Graphs saved in:", GRAPH_DIR)
