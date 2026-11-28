
import urllib.request
import zipfile
import hashlib
from pathlib import Path

# Official UCI dataset URL
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "228/sms+spam+collection.zip"
)

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"

ZIP_PATH = RAW_DIR / "sms_spam_collection.zip"
DATA_FILE = RAW_DIR / "SMSSpamCollection"


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(4096), b""):
            sha256.update(block)

    return sha256.hexdigest()


def download_dataset():

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        print("Dataset already exists.")
        print("Skipping download.")
        return

    print("Downloading official UCI SMS Spam Collection...")

    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

    print("Download completed.")

    print("Extracting dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:

        matching_files = [
            name for name in archive.namelist()
            if name.endswith("SMSSpamCollection")
        ]

        if not matching_files:
            raise FileNotFoundError(
                "SMSSpamCollection not found in downloaded archive."
            )

        DATA_FILE.write_bytes(
            archive.read(matching_files[0])
        )

    print("Extraction completed.")

    checksum = calculate_sha256(DATA_FILE)

    print("\nDataset saved at:", DATA_FILE)
    print("SHA-256 checksum:", checksum)
    print("\nDataset download successful!")


if __name__ == "__main__":
    download_dataset()
