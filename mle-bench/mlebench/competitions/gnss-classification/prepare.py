from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    # Read the GNSS training data (uses different filename)
    old_train = read_csv(raw / "GNSS_raw_train.csv")
    
    # Target column is 'Label' (0 = NLOS, 1 = LOS)
    target_col = "Label"
    id_col = "ID"
    
    # Add an ID column if it doesn't exist
    if id_col not in old_train.columns:
        old_train = old_train.reset_index()
        old_train = old_train.rename(columns={"index": id_col})
    
    # Remove rows with missing Label values (critical for grading)
    old_train = old_train.dropna(subset=[target_col])
    # Also remove rows where Label is empty string
    old_train = old_train[old_train[target_col].astype(str).str.strip() != ""]
    # Ensure Label is numeric
    old_train[target_col] = pd.to_numeric(old_train[target_col], errors='coerce')
    old_train = old_train.dropna(subset=[target_col])
    # Convert to int (0 or 1)
    old_train[target_col] = old_train[target_col].astype(int)
    
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Create sample submission
    sample_submission = new_test[[id_col]].copy()
    sample_submission["Predict"] = 1  # Default prediction (LOS)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Create private files (answers with the target column)
    new_test.to_csv(private / "test.csv", index=False)

    # Create public files visible to agents
    new_train.to_csv(public / "train.csv", index=False)

    # Create public test set without the target column
    new_test_public = new_test.drop(columns=[target_col])
    new_test_public.to_csv(public / "test.csv", index=False)

    # Checks
    assert target_col not in new_test_public.columns, "Public test set should not have target column"
    assert target_col in new_train.columns, "Train set should have target column"
    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Length of new_train and new_test should equal length of old_train"
