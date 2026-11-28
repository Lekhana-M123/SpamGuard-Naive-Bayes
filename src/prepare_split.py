
import pandas as pd

from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# ---------------------------------------
# 1. Project paths
# ---------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "SMSSpamCollection"
)

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# 2. Load original dataset
# ---------------------------------------

df = pd.read_csv(
    RAW_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    quoting=3,
    encoding="utf-8"
)

print("Original dataset size:", len(df))

# ---------------------------------------
# 3. Group-aware stratified splitting
# ---------------------------------------

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(
        X=df["message"],
        y=df["label"],
        groups=df["message"]
    )
)

train_df = df.iloc[train_indices].copy()
test_df = df.iloc[test_indices].copy()

# ---------------------------------------
# 4. Verify duplicate leakage
# ---------------------------------------

train_messages = set(train_df["message"])
test_messages = set(test_df["message"])

overlap = train_messages.intersection(test_messages)

print("\n========== SPLIT VERIFICATION ==========")

print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print("Training class distribution:")
print(train_df["label"].value_counts(normalize=True))

print("\nTesting class distribution:")
print(test_df["label"].value_counts(normalize=True))

print("\nIdentical messages in both sets:", len(overlap))

# ---------------------------------------
# 5. Save datasets
# ---------------------------------------

train_path = OUTPUT_DIR / "train.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("\nTraining data saved:", train_path)
print("Testing data saved:", test_path)

if len(overlap) == 0:
    print("\nSUCCESS: No duplicate-message leakage detected.")
else:
    raise RuntimeError(
        "Duplicate-message leakage detected!"
    )

print("\nDataset preparation completed.")
