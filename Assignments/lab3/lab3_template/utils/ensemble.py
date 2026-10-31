"""
ELEC 475: Lab 3 - Prediction Ensembles (Starter Template)
Combines the test-set predictions of several models into one prediction per image.
"""
from typing import List
import torch


def ensemble_mean(preds: List[torch.Tensor]) -> torch.Tensor:
    """
    Args:
        preds : list of 3 torch.Tensors, each shape (N, 2), float.
                Each model's predicted (u, v) in original-image pixels

    Returns:
        torch.Tensor, shape (N, 2), float. For each test image, the average of the
        three predictions, in original-image pixels (column 0 is u, column 1 is v)
    """
    return torch.stack(preds).mean(dim=0)


def ensemble_median(preds: List[torch.Tensor]) -> torch.Tensor:
    """
    Args:
        preds : list of 3 torch.Tensors, each shape (N, 2), float.
                Each model's predicted (u, v) in original-image pixels

    Returns:
        torch.Tensor, shape (N, 2), float. For each test image, the median of the
        three predictions, taken separately for u (column 0) and v (column 1),
        in original-image pixels
    """
    return torch.stack(preds).median(dim=0).values
