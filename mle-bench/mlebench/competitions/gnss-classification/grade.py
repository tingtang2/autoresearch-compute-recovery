import pandas as pd
from sklearn.metrics import accuracy_score


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Grade submission using accuracy metric.
    """
    # The target column in answers might be 'Label', submission uses 'Predict'
    target_col = "Label"
    
    merged = submission.merge(answers[["ID", target_col]], on="ID")
    
    y_pred = merged["Predict"].values
    y_true = merged[target_col].values
    
    return accuracy_score(y_true, y_pred)
