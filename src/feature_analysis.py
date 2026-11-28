
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "dataset" / "processed"
RESULT_DIR = ROOT / "results"
GRAPH_DIR = ROOT / "graphs"

RESULT_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)

# Load model training data
train_df = pd.read_csv(DATA_DIR / "train_fit.csv")

X_train = train_df["message"].fillna("")
y_train = train_df["label"]

# Convert text into word counts
vectorizer = CountVectorizer(
    stop_words=None
)

X_train_vec = vectorizer.fit_transform(X_train)

# Train model using selected alpha
model = MultinomialNB(alpha=0.1)

model.fit(X_train_vec, y_train)

# Extract class conditional log probabilities
classes = list(model.classes_)

ham_index = classes.index("ham")
spam_index = classes.index("spam")

feature_names = np.array(
    vectorizer.get_feature_names_out()
)

ham_log_prob = model.feature_log_prob_[ham_index]
spam_log_prob = model.feature_log_prob_[spam_index]

# Calculate log likelihood ratio
log_ratio = spam_log_prob - ham_log_prob

# Convert log probabilities into ordinary probabilities
ham_prob = np.exp(ham_log_prob)
spam_prob = np.exp(spam_log_prob)

# Build results table
results = pd.DataFrame({
    "word": feature_names,
    "ham_probability": ham_prob,
    "spam_probability": spam_prob,
    "log_likelihood_ratio": log_ratio
})

# Sort by strongest association
spam_words = results.nlargest(
    15,
    "log_likelihood_ratio"
)

ham_words = results.nsmallest(
    15,
    "log_likelihood_ratio"
)

print("\n========== STRONGEST SPAM WORDS ==========")

print(
    spam_words[
        ["word", "ham_probability",
         "spam_probability", "log_likelihood_ratio"]
    ].round(4).to_string(index=False)
)

print("\n========== STRONGEST HAM WORDS ==========")

print(
    ham_words[
        ["word", "ham_probability",
         "spam_probability", "log_likelihood_ratio"]
    ].round(4).to_string(index=False)
)

# Save complete feature analysis
results.to_csv(
    RESULT_DIR / "feature_analysis.csv",
    index=False
)

# Visualize strongest spam-associated words
plt.figure(figsize=(11, 7))

plt.barh(
    spam_words["word"],
    spam_words["log_likelihood_ratio"]
)

plt.xlabel("Log Likelihood Ratio")

plt.ylabel("Word")

plt.title("Words Most Strongly Associated with Spam")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "spam_word_analysis.png",
    dpi=300
)

plt.show()

# Visualize strongest ham-associated words
plt.figure(figsize=(11, 7))

plt.barh(
    ham_words["word"],
    ham_words["log_likelihood_ratio"]
)

plt.xlabel("Log Likelihood Ratio")

plt.ylabel("Word")

plt.title("Words Most Strongly Associated with Ham")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "ham_word_analysis.png",
    dpi=300
)

plt.show()

print("\nFeature analysis completed successfully.")
print("Results saved in results/feature_analysis.csv")
