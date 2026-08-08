from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    # Create train and test splits from train set
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Create sample submission
    sample_submission = new_test[["id"]].copy()
    sample_submission["diagnosed_diabetes"] = 0.5
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Create private files (answers with the target column)
    new_test.to_csv(private / "test.csv", index=False)

    # Create public files visible to agents
    new_train.to_csv(public / "train.csv", index=False)

    # Create public test set without the target column
    new_test_public = new_test.drop(columns=["diagnosed_diabetes"])
    new_test_public.to_csv(public / "test.csv", index=False)

    # Checks
    assert "diagnosed_diabetes" not in new_test_public.columns, "Public test set should not have target column"
    assert "diagnosed_diabetes" in new_train.columns, "Train set should have target column"
    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Length of new_train and new_test should equal length of old_train"
