import pandas as pd
from sklearn.metrics import accuracy_score


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Grade submission using accuracy metric.
    """
    merged = submission.merge(answers[["id", "Personality"]], on="id", suffixes=("_pred", "_true"))
    
    y_pred = merged["Personality_pred"].values
    y_true = merged["Personality_true"].values
    
    return accuracy_score(y_true, y_pred)
