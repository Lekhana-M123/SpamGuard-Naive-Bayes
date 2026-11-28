
from collections import Counter

# Small training dataset
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

# Calculate word frequencies
word_counts = {
    c: Counter() for c in classes
}

total_words = {
    c: 0 for c in classes
}

for label, message in training_data:
    words = message.split()
    word_counts[label].update(words)
    total_words[label] += len(words)

# Test different smoothing values
alpha_values = [0, 1, 5]

print("\n===== LAPLACE SMOOTHING ANALYSIS =====")

for alpha in alpha_values:

    print(f"\nSmoothing alpha = {alpha}")

    for c in classes:

        denominator = (
            total_words[c] + alpha * len(vocabulary)
        )

        print(f"\nClass: {c}")

        for word in vocabulary:

            numerator = word_counts[c][word] + alpha

            if denominator == 0 or numerator == 0:
                probability = 0
            else:
                probability = numerator / denominator

            print(
                f"P({word}|{c}) = {probability:.4f}"
            )

print("\nExperiment completed successfully.")
