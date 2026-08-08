from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    # Create train and test splits from train set
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Find target column (usually the last column or a named column)
    target_col = None
    for col in ['target', 'y', 'label', 'property']:
        if col in old_train.columns:
            target_col = col
            break
    if target_col is None:
        target_col = old_train.columns[-1]
    
    # Find id column
    id_col = None
    for col in ['id', 'Id', 'ID', 'sample_id', 'polymer_id']:
        if col in old_train.columns:
            id_col = col
            break
    if id_col is None:
        id_col = old_train.columns[0]

    # Create sample submission
    sample_submission = new_test[[id_col]].copy()
    sample_submission[target_col] = new_train[target_col].mean()  # Mean value as default
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Create private files
    new_test.to_csv(private / "test.csv", index=False)

    # Create public files visible to agents
    new_train.to_csv(public / "train.csv", index=False)
    new_test.drop(target_col, axis=1).to_csv(public / "test.csv", index=False)
