
import pandas as pd

from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# Project paths
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "dataset" / "processed"

train_path = DATA_DIR / "train.csv"

# Load existing training data
df = pd.read_csv(train_path)

print("Original training records:", len(df))

# Create training and validation partitions
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_indices, val_indices = next(
    splitter.split(
        X=df["message"],
        y=df["label"],
        groups=df["message"]
    )
)

train_fit = df.iloc[train_indices].copy()
validation = df.iloc[val_indices].copy()

# Verify duplicate separation
overlap = set(train_fit["message"]).intersection(
    set(validation["message"])
)

print("\n========== VALIDATION SPLIT ==========")

print("Model training records:", len(train_fit))
print("Validation records:", len(validation))

print("\nTraining class proportions:")
print(train_fit["label"].value_counts(normalize=True))

print("\nValidation class proportions:")
print(validation["label"].value_counts(normalize=True))

print("\nDuplicate message overlap:", len(overlap))

if len(overlap) != 0:
    raise RuntimeError("Duplicate leakage detected!")

# Save both partitions
train_fit.to_csv(
    DATA_DIR / "train_fit.csv",
    index=False
)

validation.to_csv(
    DATA_DIR / "validation.csv",
    index=False
)

print("\nTraining and validation datasets saved.")
print("Validation preparation completed successfully.")
