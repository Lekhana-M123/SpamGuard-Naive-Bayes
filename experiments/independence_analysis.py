
from collections import Counter

# Synthetic dataset for a controlled experiment
messages = [
    ("spam", "offer prize"),
    ("spam", "offer prize"),
    ("spam", "offer prize"),
    ("spam", "offer prize"),
    ("spam", "today update"),
    ("spam", "today update"),
    ("spam", "today update"),
    ("spam", "today update"),
]

target_class = "spam"

word_a = "offer"
word_b = "prize"

# Extract messages belonging to the target class
class_messages = [
    message.split()
    for label, message in messages
    if label == target_class
]

total = len(class_messages)

# Count messages containing each word
count_a = sum(
    word_a in message
    for message in class_messages
)

count_b = sum(
    word_b in message
    for message in class_messages
)

# Count messages containing both words
count_both = sum(
    word_a in message and word_b in message
    for message in class_messages
)

# Calculate individual probabilities
p_a = count_a / total
p_b = count_b / total

# Calculate actual joint probability
p_joint = count_both / total

# Calculate joint probability under independence
p_independent = p_a * p_b

print("\n===== CONDITIONAL INDEPENDENCE ANALYSIS =====")

print("Total spam messages:", total)

print(f"\nP({word_a}|spam) = {p_a:.4f}")
print(f"P({word_b}|spam) = {p_b:.4f}")

print(
    f"\nActual P({word_a}, {word_b}|spam) "
    f"= {p_joint:.4f}"
)

print(
    f"Independent estimate = "
    f"{p_independent:.4f}"
)

print(
    f"\nDifference = "
    f"{p_joint - p_independent:.4f}"
)

print("\nExperiment completed successfully.")
