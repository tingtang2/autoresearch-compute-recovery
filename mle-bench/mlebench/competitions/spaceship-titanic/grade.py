import pandas as pd
from sklearn.metrics import accuracy_score

from mlebench.competitions.utils import prepare_for_accuracy_metric
from mlebench.grade_helpers import InvalidSubmissionError


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    accuracy_inputs = prepare_for_accuracy_metric(
        submission=submission, answers=answers, target_column="Transported", id_column="PassengerId"
    )
    y_pred = pd.Series(accuracy_inputs["y_pred"])
    if y_pred.isna().any():
        raise InvalidSubmissionError(
            f"Submission contains {int(y_pred.isna().sum())} NaN predictions in `Transported`."
        )
    return accuracy_score(**accuracy_inputs)
