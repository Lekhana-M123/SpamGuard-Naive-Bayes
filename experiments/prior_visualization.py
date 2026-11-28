
import matplotlib.pyplot as plt
import numpy as np

classes = ["Ham", "Spam"]

actual_priors = [0.8660, 0.1340]
equal_priors = [0.5, 0.5]

x = np.arange(len(classes))
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    x - width / 2,
    actual_priors,
    width,
    label="Actual Dataset Priors"
)

plt.bar(
    x + width / 2,
    equal_priors,
    width,
    label="Equal Priors"
)

plt.xticks(x, classes)

plt.ylabel("Prior Probability")

plt.xlabel("Message Class")

plt.title("Comparison of Class Prior Probabilities")

plt.ylim(0, 1)

plt.legend()

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "graphs/prior_probability_comparison.png",
    dpi=300
)

plt.show()

print("Prior probability graph generated successfully.")
