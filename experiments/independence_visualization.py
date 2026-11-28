
import matplotlib.pyplot as plt

categories = [
    "Actual Joint Probability",
    "Independence Estimate"
]

probabilities = [0.50, 0.25]

plt.figure(figsize=(9, 6))

bars = plt.bar(categories, probabilities)

plt.ylabel("Probability")

plt.xlabel("Probability Calculation")

plt.title(
    "Effect of Conditional Independence Assumption"
)

plt.ylim(0, 0.65)

plt.bar_label(
    bars,
    fmt="%.2f",
    padding=5
)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "graphs/independence_comparison.png",
    dpi=300
)

plt.show()

print("Independence visualization generated successfully.")
