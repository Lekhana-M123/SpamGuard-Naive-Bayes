
import pandas as pd
from pathlib import Path

# Locate original dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "SMSSpamCollection"
)

# Read original tab-separated dataset
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    quoting=3,
    encoding="utf-8"
)

# Basic dataset inspection
print("\n========== DATASET INFORMATION ==========")

print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))

print("\nDataset shape:")
print(df.shape)

print("\nColumn information:")
print(df.info())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nClass proportions:")
print(df["label"].value_counts(normalize=True))

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate messages:")
print(df.duplicated(subset=["message"]).sum())

print("\nUnique labels:")
print(df["label"].unique())

print("\nMessage length statistics:")
print(df["message"].str.len().describe())
