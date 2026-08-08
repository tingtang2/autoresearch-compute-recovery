import pandas as pd
from sklearn.metrics import mean_absolute_error

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    # Find target column
    target_col = None
    for col in ['target', 'y', 'label', 'property']:
        if col in answers.columns:
            target_col = col
            break
    if target_col is None:
        target_col = [c for c in answers.columns if c != 'id'][-1]
    
    # Find id column
    id_col = None
    for col in ['id', 'Id', 'ID', 'sample_id', 'polymer_id']:
        if col in answers.columns:
            id_col = col
            break
    if id_col is None:
        id_col = answers.columns[0]
    
    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have '{id_col}' column")
    if target_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have '{target_col}' column")
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same length")
    
    # Sort by id to ensure correct order
    submission = submission.sort_values(by=id_col).reset_index(drop=True)
    answers = answers.sort_values(by=id_col).reset_index(drop=True)
    
    if (submission[id_col].values != answers[id_col].values).any():
        raise InvalidSubmissionError("Submission and answers must be aligned in id order")
    
    return {
        "y_true": answers[target_col].values,
        "y_pred": submission[target_col].values,
    }


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    metrics = prepare_for_metric(submission=submission, answers=answers)
    return mean_absolute_error(metrics["y_true"], metrics["y_pred"])
