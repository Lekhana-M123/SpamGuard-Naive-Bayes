
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "SMSSpamCollection"
)

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    quoting=3,
    encoding="utf-8"
)

# Find all rows belonging to repeated messages
duplicates = df[
    df.duplicated(subset="message", keep=False)
].copy()

# Group identical message texts
groups = duplicates.groupby("message")

# Count repeated unique message texts
unique_repeated_messages = len(groups)

# Identify messages associated with multiple labels
conflicting = groups.filter(
    lambda group: group["label"].nunique() > 1
)

conflicting_groups = conflicting.groupby("message")

# Display results
print("\n========== DUPLICATE ANALYSIS ==========")

print("Total dataset rows:", len(df))

print("Rows belonging to duplicate groups:", len(duplicates))

print(
    "Unique repeated message texts:",
    unique_repeated_messages
)

print(
    "Duplicate groups with conflicting labels:",
    len(conflicting_groups)
)

print("\nDuplicate label combinations:")

print(
    pd.crosstab(
        duplicates["message"],
        duplicates["label"]
    ).value_counts().head(10)
)

print("\nExamples of conflicting labels:")

if len(conflicting) > 0:
    print(
        conflicting[
            ["label", "message"]
        ].head(10).to_string(index=False)
    )
else:
    print("No conflicting duplicate labels found.")

print("\nDuplicate analysis completed.")
