import pandas as pd
from sklearn.metrics import cohen_kappa_score


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Grade submission using quadratic weighted kappa metric.
    """
    # Normalize column names for case sensitivity
    # Ensure submission has 'Id' (not 'id')
    if "Id" not in submission.columns and "id" in submission.columns:
        submission = submission.rename(columns={"id": "Id"})
    
    # Ensure answers has 'Id' (not 'id')
    if "Id" not in answers.columns and "id" in answers.columns:
        answers = answers.rename(columns={"id": "Id"})
    
    # Ensure both have required columns
    if "Id" not in submission.columns:
        raise ValueError(f"Submission missing 'Id' column. Found columns: {list(submission.columns)}")
    if "quality" not in submission.columns:
        raise ValueError(f"Submission missing 'quality' column. Found columns: {list(submission.columns)}")
    if "Id" not in answers.columns:
        raise ValueError(f"Answers missing 'Id' column. Found columns: {list(answers.columns)}")
    if "quality" not in answers.columns:
        raise ValueError(f"Answers missing 'quality' column. Found columns: {list(answers.columns)}")
    
    merged = submission.merge(answers[["Id", "quality"]], on="Id", suffixes=("_pred", "_true"))
    
    y_pred = merged["quality_pred"].values
    y_true = merged["quality_true"].values
    
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")
