import numpy as np
import pandas as pd


def apk(actual: str, predicted: str, k: int = 3) -> float:
    """
    Compute Average Precision at K for a single sample.
    """
    if not predicted:
        return 0.0
    
    predicted_list = predicted.split()[:k]
    
    score = 0.0
    num_hits = 0.0
    
    for i, p in enumerate(predicted_list):
        if p == actual and p not in predicted_list[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    return score


def mapk(actuals: list, predictions: list, k: int = 3) -> float:
    """
    Compute Mean Average Precision at K.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actuals, predictions)])


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Grade submission using MAP@3 metric.
    """
    merged = submission.merge(answers, on="id", suffixes=("_pred", "_true"))
    
    pred_col = [c for c in merged.columns if c.endswith("_pred")][0]
    true_col = [c for c in merged.columns if c.endswith("_true")][0]
    
    predictions = merged[pred_col].fillna("").astype(str).tolist()
    actuals = merged[true_col].astype(str).tolist()
    
    return mapk(actuals, predictions, k=3)
