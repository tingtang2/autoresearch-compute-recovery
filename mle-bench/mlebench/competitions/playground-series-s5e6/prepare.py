from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    # Create train and test splits from train set
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Identify the target column (Fertilizer Name)
    target_col = "Fertilizer Name"

    # Create sample submission
    sample_submission = new_test[["id"]].copy()
    sample_submission[target_col] = "14-35-14 10-26-26 Urea"  # Default prediction
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
