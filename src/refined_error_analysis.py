
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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
# 3. Train Naive Bayes model
# -----------------------------------------

vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=0.1)
model.fit(X_train_vec, y_train)

spam_index = list(model.classes_).index("spam")

probabilities = model.predict_proba(
    X_val_vec
)[:, spam_index]

y_pred = (probabilities >= 0.5).astype(int)

# -----------------------------------------
# 4. Identify errors
# -----------------------------------------

error_df = pd.DataFrame({
    "message": X_val.to_numpy(),
    "actual": y_true,
    "predicted": y_pred,
    "spam_probability": probabilities
})

fp = error_df[
    (error_df["actual"] == 0) &
    (error_df["predicted"] == 1)
].copy()

fn = error_df[
    (error_df["actual"] == 1) &
    (error_df["predicted"] == 0)
].copy()

print("\n========== REFINED ERROR ANALYSIS ==========")

print("False Positives:", len(fp))
print("False Negatives:", len(fn))

# -----------------------------------------
# 5. Prepare meaningful stopwords
# -----------------------------------------

# Preserve important negation words.
preserved_words = {"no", "not", "nor", "never"}

custom_stopwords = sorted(
    set(ENGLISH_STOP_WORDS) - preserved_words
)

# -----------------------------------------
# 6. Extract meaningful words and phrases
# -----------------------------------------

def extract_terms(data, error_type):

    if len(data) == 0:
        print(f"\nNo {error_type} examples found.")
        return []

    texts = data["message"].astype(str).tolist()

    term_vectorizer = CountVectorizer(
        stop_words=custom_stopwords,
        ngram_range=(1, 2),
        binary=True,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
    )

    matrix = term_vectorizer.fit_transform(texts)

    terms = term_vectorizer.get_feature_names_out()

    document_counts = matrix.sum(axis=0).A1

    total_messages = len(data)

    results = []

    for term, count in zip(terms, document_counts):

        results.append({
            "error_type": error_type,
            "term": term,
            "message_count": int(count),
            "percentage_of_messages": (
                count / total_messages
            ) * 100
        })

    results.sort(
        key=lambda x: x["message_count"],
        reverse=True
    )

    return results


fp_terms = extract_terms(fp, "False Positive")
fn_terms = extract_terms(fn, "False Negative")

# -----------------------------------------
# 7. Combine results
# -----------------------------------------

all_terms = fp_terms + fn_terms

terms_df = pd.DataFrame(all_terms)

if not terms_df.empty:

    terms_df.to_csv(
        RESULT_DIR / "error_terms_by_type.csv",
        index=False
    )

    print("\n========== FALSE POSITIVE TERMS ==========")

    print(
        terms_df[
            terms_df["error_type"] == "False Positive"
        ].head(15).to_string(index=False)
    )

    print("\n========== FALSE NEGATIVE TERMS ==========")

    print(
        terms_df[
            terms_df["error_type"] == "False Negative"
        ].head(15).to_string(index=False)
    )

# -----------------------------------------
# 8. Visualize meaningful error terms
# -----------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 7)
)

groups = [
    ("False Positive", axes[0]),
    ("False Negative", axes[1])
]

for error_type, ax in groups:

    subset = terms_df[
        terms_df["error_type"] == error_type
    ].head(10)

    if subset.empty:
        ax.set_title(f"{error_type}: No errors")
        continue

    subset = subset.sort_values("message_count")

    ax.barh(
        subset["term"],
        subset["message_count"]
    )

    ax.set_title(f"Top Terms: {error_type}")
    ax.set_xlabel("Number of Error Messages")
    ax.set_ylabel("Word / Phrase")

    ax.grid(axis="x", alpha=0.3)

plt.suptitle(
    "Meaningful Words and Phrases in Classification Errors"
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "refined_error_terms.png",
    dpi=300
)

plt.show()

# -----------------------------------------
# 9. Final summary
# -----------------------------------------

print("\n========== OUTPUT FILES ==========")

print(
    "Term analysis:",
    RESULT_DIR / "error_terms_by_type.csv"
)

print(
    "Visualization:",
    GRAPH_DIR / "refined_error_terms.png"
)

print("\nRefined error analysis completed successfully.")
