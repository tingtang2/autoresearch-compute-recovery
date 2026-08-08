import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Grade submission using multi-class log loss metric.
    """
    # Merge on id
    merged = submission.merge(answers[["id", "Status"]], on="id")
    
    # Get predicted probabilities
    prob_cols = ["Status_C", "Status_CL", "Status_D"]
    y_pred = merged[prob_cols].values
    
    # Normalize probabilities (each row sums to 1)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    
    # Clip probabilities to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Create one-hot encoded true labels
    classes = ["C", "CL", "D"]
    y_true = merged["Status"].values
    
    # Calculate log loss
    return log_loss(y_true, y_pred, labels=classes)
