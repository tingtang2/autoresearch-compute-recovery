import pandas as pd
from sklearn.metrics import mean_absolute_error

from mlebench.grade_helpers import InvalidSubmissionError


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """Grade demand forecasting submission using MAE (regression metric)."""
    id_col = None
    target_col = None

    for col in ["id", "Id", "ID", "date", "timestamp", "time", "datetime", "index"]:
        if col in answers.columns:
            id_col = col
            break
    if id_col is None:
        id_col = answers.columns[0]

    for col in ["demand", "electricity_demand", "consumption", "value", "target", "y", "class"]:
        if col in answers.columns:
            target_col = col
            break
    if target_col is None:
        target_col = [c for c in answers.columns if c != id_col][-1]

    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission must contain the '{id_col}' column.")
    if target_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission must contain the '{target_col}' column.")

    submission = submission.sort_values(by=id_col).reset_index(drop=True)
    answers = answers.sort_values(by=id_col).reset_index(drop=True)

    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same length.")
    if (submission[id_col].values != answers[id_col].values).any():
        raise InvalidSubmissionError("Submission and answers must be aligned by id.")

    y_true = pd.to_numeric(answers[target_col], errors="coerce")
    y_pred = pd.to_numeric(submission[target_col], errors="coerce")
    if y_true.isna().any() or y_pred.isna().any():
        raise InvalidSubmissionError(
            "Target column must be numeric for regression (demand forecasting)."
        )

    return float(mean_absolute_error(y_true, y_pred))
