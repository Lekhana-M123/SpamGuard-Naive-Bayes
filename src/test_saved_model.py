
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

# Load saved files
model = joblib.load(
    MODEL_DIR / "naive_bayes_model.pkl"
)

vectorizer = joblib.load(
    MODEL_DIR / "count_vectorizer.pkl"
)

config = joblib.load(
    MODEL_DIR / "model_config.pkl"
)

threshold = config["decision_threshold"]

# Test messages
test_messages = [
    "Congratulations! You won a free prize. Call now!",
    "Hey, are we meeting tomorrow?",
    "URGENT! Claim your free cash reward now!",
    "Please call me when you reach home.",
    "You have won a lottery. Send your details.",
    "I will meet you at the railway station."
]

print("\n========== SAVED MODEL VERIFICATION ==========")

for message in test_messages:

    message_vector = vectorizer.transform([message])

    probabilities = model.predict_proba(message_vector)[0]

    spam_index = list(model.classes_).index("spam")

    spam_probability = probabilities[spam_index]

    prediction = (
        "SPAM"
        if spam_probability >= threshold
        else "HAM"
    )

    print("\nMessage:", message)
    print("Prediction:", prediction)
    print(f"Spam probability: {spam_probability * 100:.2f}%")

print("\nSaved model verification completed.")
