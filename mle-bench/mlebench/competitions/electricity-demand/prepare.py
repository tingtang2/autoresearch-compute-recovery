from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    # Read the training data
    # The dataset might have different file structures, so we try common patterns
    train_file = None
    for filename in ['train.csv', 'data.csv', 'electricity_demand.csv', 'electricity.csv']:
        if (raw / filename).exists():
            train_file = raw / filename
            break
    
    # If no standard file found, try to find any CSV file
    if train_file is None:
        csv_files = list(raw.glob("*.csv"))
        if csv_files:
            train_file = csv_files[0]
    
    if train_file is None:
        raise FileNotFoundError(f"Could not find training data in {raw}")
    
    old_train = read_csv(train_file)
    
    # Create train and test splits from train set
    # For time series, we might want to split chronologically, but for simplicity,
    # we'll use random split. If the data has a time column, it should be preserved.
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Find target column (class, demand, electricity_demand, consumption, value, target, y, label)
    target_col = None
    for col in ['class', 'demand', 'electricity_demand', 'consumption', 'value', 'target', 'y', 'label']:
        if col in old_train.columns:
            target_col = col
            break
    if target_col is None:
        # Use the last column as target
        target_col = old_train.columns[-1]
    
    # Find id column (might be date, timestamp, id, or index)
    # For this dataset, we'll create a sequential id
    id_col = 'id'
    if id_col not in old_train.columns:
        # Add an id column based on index
        new_train[id_col] = range(len(new_train))
        new_test[id_col] = range(len(new_train), len(new_train) + len(new_test))
    
    # Create sample submission
    sample_submission = new_test[[id_col]].copy()
    # Use most frequent class as default prediction (classification)
    most_frequent = new_train[target_col].mode()[0]
    # Clean up the value if it has 'b' prefix (from bytes encoding)
    if isinstance(most_frequent, str) and most_frequent.startswith("b'") and most_frequent.endswith("'"):
        most_frequent = most_frequent[2:-1]
    sample_submission[target_col] = most_frequent
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Create private files
    new_test.to_csv(private / "test.csv", index=False)

    # Create public files visible to agents
    new_train.to_csv(public / "train.csv", index=False)
    # Remove target column from public test
    new_test_without_target = new_test.drop(target_col, axis=1, errors='ignore')
    new_test_without_target.to_csv(public / "test.csv", index=False)
