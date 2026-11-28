
import matplotlib.pyplot as plt

alpha_values = [0, 0.5, 1, 2, 5, 10, 20]

# Word counts from our experiment
ham_counts = {
    "free": 0,
    "meeting": 2,
    "now": 2
}

spam_counts = {
    "free": 4,
    "meeting": 0,
    "now": 2
}

ham_total = 4
spam_total = 6
vocab_size = 3

words = ["free", "meeting", "now"]

plt.figure(figsize=(12, 6))

for word in words:

    ham_probabilities = []
    spam_probabilities = []

    for alpha in alpha_values:

        ham_p = (
            ham_counts[word] + alpha
        ) / (
            ham_total + alpha * vocab_size
        )

        spam_p = (
            spam_counts[word] + alpha
        ) / (
            spam_total + alpha * vocab_size
        )

        ham_probabilities.append(ham_p)
        spam_probabilities.append(spam_p)

    plt.plot(
        alpha_values,
        ham_probabilities,
        marker="o",
        label=f"Ham - {word}"
    )

    plt.plot(
        alpha_values,
        spam_probabilities,
        marker="s",
        linestyle="--",
        label=f"Spam - {word}"
    )

plt.xlabel("Smoothing Parameter (Alpha)")
plt.ylabel("Estimated Word Probability")

plt.title(
    "Effect of Laplace Smoothing on Word Probabilities"
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "graphs/smoothing_probability_analysis.png",
    dpi=300
)

plt.show()

print("Smoothing visualization generated successfully.")
