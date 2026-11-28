
from collections import Counter

# ---------------------------------------
# 1. Small artificial training dataset
# ---------------------------------------

training_data = [
    ("ham", "meeting now"),
    ("ham", "meeting"),
    ("ham", "now"),
    ("spam", "free now"),
    ("spam", "free free"),
    ("spam", "free now")
]

vocabulary = ["free", "meeting", "now"]

classes = ["ham", "spam"]

alpha = 1

# ---------------------------------------
# 2. Calculate class priors
# ---------------------------------------

total_messages = len(training_data)

class_counts = Counter(
    label for label, message in training_data
)

priors = {
    c: class_counts[c] / total_messages
    for c in classes
}

print("\n========== CLASS PRIORS ==========")

for c in classes:
    print(f"P({c}) = {priors[c]:.4f}")

# ---------------------------------------
# 3. Calculate word frequencies
# ---------------------------------------

word_counts = {
    c: Counter()
    for c in classes
}

total_words = {
    c: 0
    for c in classes
}

for label, message in training_data:

    words = message.split()

    word_counts[label].update(words)

    total_words[label] += len(words)

print("\n========== WORD COUNTS ==========")

for c in classes:

    print(f"\nClass: {c}")

    for word in vocabulary:
        print(f"{word}: {word_counts[c][word]}")

# ---------------------------------------
# 4. Calculate smoothed likelihoods
# ---------------------------------------

likelihoods = {
    c: {}
    for c in classes
}

print("\n========== WORD LIKELIHOODS ==========")

for c in classes:

    denominator = (
        total_words[c] + alpha * len(vocabulary)
    )

    for word in vocabulary:

        probability = (
            word_counts[c][word] + alpha
        ) / denominator

        likelihoods[c][word] = probability

        print(
            f"P({word}|{c}) = {probability:.4f}"
        )

# ---------------------------------------
# 5. Predict a new message
# ---------------------------------------

test_message = "free now"

test_words = test_message.split()

scores = {}

for c in classes:

    score = priors[c]

    for word in test_words:

        score *= likelihoods[c][word]

    scores[c] = score

# ---------------------------------------
# 6. Normalize posterior probabilities
# ---------------------------------------

total_score = sum(scores.values())

posteriors = {
    c: scores[c] / total_score
    for c in classes
}

print("\n========== PREDICTION ==========")

for c in classes:

    print(f"{c}: {posteriors[c] * 100:.2f}%")

predicted_class = max(
    posteriors,
    key=posteriors.get
)

print("\nTest message:", test_message)

print("Predicted class:", predicted_class)

print("\nExperiment completed successfully.")
